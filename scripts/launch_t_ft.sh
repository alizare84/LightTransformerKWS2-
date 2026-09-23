#!/usr/bin/env bash
# Launch T+2K FT control, logging to results/t_ft_control_run.log
# Uses anaconda3 base env which has torch+cuda

PYTHON=/home/alizare84/anaconda3/bin/python
cd /home/alizare84/LightTransformerKWS2
mkdir -p results

LOG=results/t_ft_control_run.log
echo "===== $(date '+%a %b %d %H:%M:%S %Z %Y') T+2K FT control start =====" > "$LOG"

$PYTHON scripts/run_t_ft_control.py \
    --seeds 0,1,2,3,4 \
    --steps 2000 \
    --lr 1e-4 \
    --device cuda \
    --num-workers 2 \
    >> "$LOG" 2>&1

echo "===== $(date '+%a %b %d %H:%M:%S %Z %Y') T+2K FT control done =====" >> "$LOG"
