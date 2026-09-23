#!/usr/bin/env python3
"""Launch run_t_ft_control.py as a detached subprocess that survives WSL session end."""
import os
import sys
import subprocess

PROJECT = '/home/alizare84/LightTransformerKWS2'
PYTHON  = '/home/alizare84/anaconda3/bin/python'
SCRIPT  = os.path.join(PROJECT, 'scripts', 'run_t_ft_control.py')
LOG_OUT = os.path.join(PROJECT, 'results', 't_ft_control_run.log')
LOG_ERR = os.path.join(PROJECT, 'results', 't_ft_control_err.log')

os.makedirs(os.path.join(PROJECT, 'results'), exist_ok=True)

# Clear old logs
for p in (LOG_OUT, LOG_ERR):
    open(p, 'w').close()

cmd = [PYTHON, '-u', SCRIPT,
       '--seeds', '0,1,2,3,4',
       '--steps', '2000',
       '--lr',    '1e-4',
       '--device', 'cuda']

with open(LOG_OUT, 'w') as fout, open(LOG_ERR, 'w') as ferr:
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT,
        stdout=fout,
        stderr=ferr,
        # detach from process group so it survives parent exit
        start_new_session=True,
    )

print(f"Launched PID={proc.pid}")
print(f"stdout -> {LOG_OUT}")
print(f"stderr -> {LOG_ERR}")
