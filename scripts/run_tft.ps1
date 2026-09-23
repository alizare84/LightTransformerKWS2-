# PowerShell script to launch T+2K FT control via WSL
# Uses updated run_t_ft_control.py that calls Datagen directly (no worker subprocesses)
$logOut = '\\wsl.localhost\Ubuntu-20.04\home\alizare84\LightTransformerKWS2\results\t_ft_control_run.log'
$logErr = '\\wsl.localhost\Ubuntu-20.04\home\alizare84\LightTransformerKWS2\results\t_ft_control_err.log'

# Remove old logs
Remove-Item $logOut -ErrorAction SilentlyContinue
Remove-Item $logErr -ErrorAction SilentlyContinue

$proc = Start-Process -PassThru `
    -FilePath 'wsl.exe' `
    -ArgumentList @(
        '-d', 'Ubuntu-20.04',
        'bash', '-c',
        'cd /home/alizare84/LightTransformerKWS2 && /home/alizare84/anaconda3/bin/python -u scripts/run_t_ft_control.py --seeds 0,1,2,3,4 --steps 2000 --lr 1e-4 --device cuda'
    ) `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -NoNewWindow

Write-Host "T+2K FT started with PID: $($proc.Id)"
Write-Host "stdout log: $logOut"
Write-Host "stderr log: $logErr"
