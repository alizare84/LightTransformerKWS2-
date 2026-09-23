#!/usr/bin/env bash
# Find python with torch
echo "=== Searching for python with torch ==="
for p in /usr/bin/python3 /usr/local/bin/python3 $(find /opt /root /home/alizare84 -maxdepth 8 -name 'python3' -path '*/bin/*' 2>/dev/null | head -20); do
    if [ -x "$p" ]; then
        result=$("$p" -c "import torch; print('$p torch=' + torch.__version__)" 2>/dev/null)
        if [ -n "$result" ]; then
            echo "FOUND: $result"
        fi
    fi
done

echo "=== conda envs ==="
conda env list 2>/dev/null || echo "conda not in PATH"

echo "=== PATH python ==="
which python python3 2>/dev/null
python3 -c "import sys; print(sys.executable, sys.version)" 2>/dev/null

echo "=== train.sh content ==="
cat /home/alizare84/LightTransformerKWS2/train.sh 2>/dev/null

echo "=== bash profile ==="
cat /home/alizare84/.bashrc 2>/dev/null | grep -i 'conda\|python\|PATH\|activate' | head -20
