import argparse
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.agent import (
    OpponentsCfg,
    SaveHandler,
    SaveHandlerMode,
    SelfPlayRandom,
    TrainLogging,
    CameraResolution,
    train,
    ConstantAgent,
    BasedAgent,
)

from user.train_agent import CustomAgent, MLPExtractor, gen_reward_manager
from environment.agent import run_match, plot_results
import os
import re


def opponent_mix(progress: float, selfplay_handler: SelfPlayRandom):
    """
    Curriculum schedule for opponent probabilities.
    progress: 0.0 -> 1.0 across the 80M plan
    Returns a dict suitable for OpponentsCfg(opponents=...)
    """
    if progress < 0.10:
        # Warm-up: learn to move/attack/recover
        return {
            'constant_agent': (2.0, partial(ConstantAgent)),
            'based_agent': (2.0, partial(BasedAgent)),
            'self_play': (1.0, selfplay_handler),
        }
    elif progress < 0.30:
        # Start mixing in self-play
        return {
            'constant_agent': (1.0, partial(ConstantAgent)),
            'based_agent': (2.0, partial(BasedAgent)),
            'self_play': (4.0, selfplay_handler),
        }
    elif progress < 0.70:
        # Self-play heavy
        return {
            'constant_agent': (0.5, partial(ConstantAgent)),
            'based_agent': (1.5, partial(BasedAgent)),
            'self_play': (6.0, selfplay_handler),
        }
    else:
        # Mostly self-play for polish
        return {
            'constant_agent': (0.2, partial(ConstantAgent)),
            'based_agent': (0.8, partial(BasedAgent)),
            'self_play': (8.0, selfplay_handler),
        }


def main():
    parser = argparse.ArgumentParser(description="Automate AI^2 training to large timesteps.")
    parser.add_argument("--run-name", default="comp_run_80m", help="Folder name under checkpoints/")
    parser.add_argument("--save-path", default="checkpoints", help="Base path for checkpoints")
    parser.add_argument("--total-steps", type=int, default=80_000_000, help="Total training timesteps")
    parser.add_argument("--segment-steps", type=int, default=2_000_000, help="Steps per training segment")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint frequency")
    parser.add_argument("--max-saved", type=int, default=50, help="Max number of checkpoints to keep (-1 = unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing run directory (RESUME mode)")
    parser.add_argument("--log", choices=["none", "file", "plot"], default="none", help="Training logs level")
    parser.add_argument("--vec-envs", type=int, default=1, help="Number of parallel envs (use >1 to leverage GPU batches)")
    parser.add_argument("--vec-type", choices=["subproc", "dummy"], default="subproc", help="VecEnv type")
    parser.add_argument("--no-eval", action="store_true", help="Disable post-segment MP4 evaluations")
    parser.add_argument("--eval-seconds", type=int, default=30, help="Seconds per evaluation video")
    parser.add_argument("--eval-opponent", choices=["constant", "based"], default="based", help="Opponent used during evaluation videos")
    parser.add_argument("--selfplay", choices=["random", "latest", "off"], default="random", help="Self-play mode for single-env segments")
    parser.add_argument("--selfplay-weight", type=float, default=1.0, help="Relative weight for self-play opponent in single-env segments")
    args = parser.parse_args()

    def update_eval_index(eval_dir: str, run_name: str):
        try:
            files = [f for f in os.listdir(eval_dir) if f.lower().endswith('.mp4')]
            def sort_key(name: str):
                m = re.search(r"seg_(\d+)_eval\.mp4$", name)
                return int(m.group(1)) if m else 0
            files.sort(key=sort_key)
            items = []
            for f in files:
                step = sort_key(f)
                items.append(f"<li><a href=\"{f}\">Segment {step:,} steps</a></li>")
            html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>AI^2 Eval Videos - {run_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ margin-bottom: 0.5rem; }}
    ul {{ line-height: 1.7; }}
  </style>
  <meta http-equiv=\"Cache-Control\" content=\"no-cache, no-store, must-revalidate\" />
  <meta http-equiv=\"Pragma\" content=\"no-cache\" />
  <meta http-equiv=\"Expires\" content=\"0\" />
  <meta http-equiv=\"refresh\" content=\"15\"> <!-- auto-refresh every 15s -->
  </head>
<body>
  <h1>Evaluation Videos - {run_name}</h1>
  <p>Newest at bottom. Auto-refreshes every 15s.</p>
  <ul>
    {''.join(items) if items else '<li>No videos yet.</li>'}
  </ul>
</body>
</html>
"""
            with open(os.path.join(eval_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            # Non-fatal
            pass

    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Save handler (RESUME to keep pool for self-play)
    mode = SaveHandlerMode.RESUME if args.resume else SaveHandlerMode.FORCE
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=args.save_freq,
        max_saved=args.max_saved,
        save_path=args.save_path,
        run_name=args.run_name,
        mode=mode,
    )

    # Train in segments so we can adjust curriculum and resume safely
    steps_done = 0
    while steps_done < args.total_steps:
        seg_steps = min(args.segment_steps, args.total_steps - steps_done)
        progress = steps_done / max(args.total_steps, 1)

        # Map log arg
        log_map = {
            "none": TrainLogging.NONE,
            "file": TrainLogging.TO_FILE,
            "plot": TrainLogging.PLOT,
        }

        # If vectorized, avoid self-play inside env processes; use constant/based for speed
        if args.vec_envs > 1:
            from environment.agent import SelfPlayWarehouseBrawl

            def make_env(opponents_dict):
                def _init():
                    # Per-env reward manager
                    rm = gen_reward_manager()
                    cfg = OpponentsCfg(opponents=opponents_dict)
                    env = SelfPlayWarehouseBrawl(
                        reward_manager=rm,
                        opponent_cfg=cfg,
                        save_handler=None,
                        resolution=CameraResolution.LOW,
                    )
                    # Ensure signals are connected
                    rm.subscribe_signals(env.raw_env)
                    return env
                return _init

            # Curriculum without self-play for vec training
            if progress < 0.10:
                opp_spec = {
                    'constant_agent': (2.0, partial(ConstantAgent)),
                    'based_agent': (2.0, partial(BasedAgent)),
                }
            else:
                opp_spec = {
                    'constant_agent': (1.0, partial(ConstantAgent)),
                    'based_agent': (2.0, partial(BasedAgent)),
                }

            env_fns = [make_env(opp_spec) for _ in range(args.vec_envs)]
            vec_cls = SubprocVecEnv if args.vec_type == "subproc" else DummyVecEnv
            vec_env = vec_cls(env_fns)

            # Optional logging for learning curve
            log_dir = f"{save_handler._experiment_path()}/"
            if args.log != "none":
                os.makedirs(log_dir, exist_ok=True)
                vec_env = VecMonitor(vec_env, log_dir)

            # VecNormalize: normalize obs/rewards for more stable/efficient learning
            norm_path = os.path.join(log_dir, "vecnorm.pkl")

            if os.path.isfile(norm_path):
                 vec_env = VecNormalize.load(norm_path, vec_env)
                 vec_env.training = True
                 vec_env.norm_reward = True
            else:
                 vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            
            def save_norm_stats():
                 try:
                     vec_env.save(norm_path)
                 except Exception as e:
                     print(f"couldn't save vecNormalize stats: {e}")

            # Periodic saver callback (main process)
            class PeriodicSaver(BaseCallback):
                def __init__(self, save_handler, check_freq):
                    super().__init__()
                    self.save_handler = save_handler
                    self.check_freq = check_freq
                    self._last = 0

                def _on_step(self) -> bool:
                    # Save when crossing next multiple of check_freq
                    t = self.num_timesteps
                    if t - self._last >= self.check_freq:
                        self._last = t
                        self.save_handler.update_info()
                        self.save_handler.save_agent()
                    return True

            saver_cb = PeriodicSaver(save_handler, args.save_freq)

            # Initialize model on first segment using a single base env
            if not getattr(my_agent, "initialized", False):
                base_rm = gen_reward_manager()
                base_cfg = OpponentsCfg(opponents=opp_spec)
                base_env = SelfPlayWarehouseBrawl(
                    reward_manager=base_rm,
                    opponent_cfg=base_cfg,
                    save_handler=None,
                    resolution=CameraResolution.LOW,
                )
                base_rm.subscribe_signals(base_env.raw_env)
                my_agent.get_env_info(base_env)
                # Release the base env; model is initialized now
                try:
                    base_env.close()
                except Exception:
                    pass

            my_agent.learn(
                env=vec_env,
                total_timesteps=seg_steps,
                log_interval=1,
                verbose=0,
                callback=saver_cb,
            )

            save_norm_stats()
            
            vec_env.close()
            
            # Save normalization stats for next segment reuse
            try:
                vec_env.save(norm_path)
            except Exception:
                pass
            vec_env.close()

            # Plot learning curve if requested
            if args.log == "plot":
                try:
                    plot_results(log_dir, file_suffix=f"_{steps_done + seg_steps}")
                except Exception:
                    pass

            # Evaluation video after this segment
            if not args.no_eval:
                eval_dir = os.path.join("results", args.run_name)
                os.makedirs(eval_dir, exist_ok=True)
                opponent = BasedAgent if args.eval_opponent == "based" else ConstantAgent
                outfile = os.path.join(eval_dir, f"seg_{steps_done + seg_steps}_eval.mp4")
                # Short, deterministic-ish eval
                reward_manager = gen_reward_manager()
                try:
                    run_match(
                        my_agent,
                        agent_2=opponent,
                        video_path=outfile,
                        agent_1_name='Agent',
                        agent_2_name=args.eval_opponent.capitalize(),
                        resolution=CameraResolution.LOW,
                        reward_manager=reward_manager,
                        max_timesteps=30 * args.eval_seconds,
                        train_mode=False,
                    )
                    update_eval_index(eval_dir, args.run_name)
                except Exception:
                    # Don't break training if eval fails
                    pass
        else:
            # Single-env path with self-play and SaveHandler integration
            # Configure self-play mode
            if args.selfplay == "latest":
                from environment.agent import SelfPlayLatest
                selfplay_handler = SelfPlayLatest(partial(type(my_agent)))
            elif args.selfplay == "off":
                selfplay_handler = None
            else:
                selfplay_handler = SelfPlayRandom(partial(type(my_agent)))

            if selfplay_handler is None:
                opp_spec = {
                    'constant_agent': (2.0, partial(ConstantAgent)),
                    'based_agent': (3.0, partial(BasedAgent)),
                }
            else:
                opp_spec = opponent_mix(progress, selfplay_handler)
                if 'self_play' in opp_spec and isinstance(opp_spec['self_play'], tuple):
                    _, handler = opp_spec['self_play']
                    opp_spec['self_play'] = (args.selfplay_weight, handler)
            opponent_cfg = OpponentsCfg(opponents=opp_spec)

            # Build a DummyVecEnv + VecMonitor + VecNormalize for single-env training
            from environment.agent import SelfPlayWarehouseBrawl

            def make_env_single():
                def _init():
                    rm = gen_reward_manager()
                    env = SelfPlayWarehouseBrawl(
                        reward_manager=rm,
                        opponent_cfg=opponent_cfg,
                        save_handler=save_handler,
                        resolution=CameraResolution.LOW,
                    )
                    rm.subscribe_signals(env.raw_env)
                    return env
                return _init

            log_dir = f"{save_handler._experiment_path()}/"
            os.makedirs(log_dir, exist_ok=True)
            env_fns = [make_env_single()]
            vec_env = DummyVecEnv(env_fns)
            vec_env = VecMonitor(vec_env, log_dir)

            # Normalize obs/rewards and persist stats
            norm_path = os.path.join(log_dir, "vecnorm_single.pkl")
            if os.path.isfile(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = True
            else:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            # Initialize model on first segment using a base env
            if not getattr(my_agent, "initialized", False):
                base_rm = gen_reward_manager()
                base_env = SelfPlayWarehouseBrawl(
                    reward_manager=base_rm,
                    opponent_cfg=opponent_cfg,
                    save_handler=save_handler,
                    resolution=CameraResolution.LOW,
                )
                base_rm.subscribe_signals(base_env.raw_env)
                my_agent.get_env_info(base_env)
                try:
                    base_env.close()
                except Exception:
                    pass

            # Periodic saver
            class PeriodicSaver(BaseCallback):
                def __init__(self, save_handler, check_freq):
                    super().__init__()
                    self.save_handler = save_handler
                    self.check_freq = check_freq
                    self._last = 0

                def _on_step(self) -> bool:
                    t = self.num_timesteps
                    if t - self._last >= self.check_freq:
                        self._last = t
                        self.save_handler.update_info()
                        self.save_handler.save_agent()
                    return True

            saver_cb = PeriodicSaver(save_handler, args.save_freq)

            # Initialize save handler counters before env starts stepping
            try:
                save_handler.update_info()
            except Exception:
                pass

            my_agent.learn(
                env=vec_env,
                total_timesteps=seg_steps,
                log_interval=1,
                verbose=0,
                callback=saver_cb,
            )

            try:
                vec_env.save(norm_path)
            except Exception:
                pass
            vec_env.close()

            # Plot learning curve if requested
            if args.log == "plot":
                try:
                    plot_results(log_dir, file_suffix=f"_{steps_done + seg_steps}")
                except Exception:
                    pass

            # Evaluation video after this segment
            if not args.no_eval:
                eval_dir = os.path.join("results", args.run_name)
                os.makedirs(eval_dir, exist_ok=True)
                opponent = BasedAgent if args.eval_opponent == "based" else ConstantAgent
                outfile = os.path.join(eval_dir, f"seg_{steps_done + seg_steps}_eval.mp4")
                reward_manager = gen_reward_manager()
                try:
                    run_match(
                        my_agent,
                        agent_2=opponent,
                        video_path=outfile,
                        agent_1_name='Agent',
                        agent_2_name=args.eval_opponent.capitalize(),
                        resolution=CameraResolution.LOW,
                        reward_manager=reward_manager,
                        max_timesteps=30 * args.eval_seconds,
                        train_mode=False,
                    )
                    update_eval_index(eval_dir, args.run_name)
                except Exception:
                    pass

        steps_done += seg_steps


if __name__ == "__main__":
    main()


