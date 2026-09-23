#!/usr/bin/env bash
# Wrapper script to run T+2K FT control from inside WSL
# Logs to results/ from within WSL (avoids Windows path redirect issues)

LOG=/home/alizare84/LightTransformerKWS2/results/t_ft_control_run.log
ERRLOG=/home/alizare84/LightTransformerKWS2/results/t_ft_control_err.log

cd /home/alizare84/LightTransformerKWS2
mkdir -p results

rm -f "$LOG" "$ERRLOG"

nohup /home/alizare84/anaconda3/bin/python -u \
    scripts/run_t_ft_control.py \
    --seeds 0,1,2,3,4 \
    --steps 2000 \
    --lr 1e-4 \
    --device cuda \
    > "$LOG" 2> "$ERRLOG" &

echo "T+2K FT launched, PID=$!, log=$LOG"
