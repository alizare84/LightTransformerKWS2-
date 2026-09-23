# run_tft2.ps1 - relaunch T+2K FT control with latest script
$logOut = '\\wsl.localhost\Ubuntu-20.04\home\alizare84\LightTransformerKWS2\results\t_ft_control_run.log'
$logErr = '\\wsl.localhost\Ubuntu-20.04\home\alizare84\LightTransformerKWS2\results\t_ft_control_err.log'

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

Write-Host "T+2K FT relaunched PID=$($proc.Id)"
