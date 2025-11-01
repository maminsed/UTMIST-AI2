# Usage: .\scripts\train_80m.ps1 [-RunName comp_run_80m]

param(
  [string]$RunName = "comp_run_80m"
)

$env:PYGAME_HIDE_SUPPORT_PROMPT = "1"

python scripts/auto_train.py `
  --run-name $RunName `
  --save-path checkpoints `
  --total-steps 80000000 `
  --segment-steps 2000000 `
  --save-freq 50000 `
  --max-saved 50 `
  --resume `
  --log none `
  --vec-envs 8 `
  --vec-type subproc
