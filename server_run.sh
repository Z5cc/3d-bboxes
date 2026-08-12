#!/usr/bin/env bash

source ./config.sh

# copy python scripts to server
rsync --rsh "ssh -p $PORT" --info=progress2      *.py $HOST:/workspace/

# train
ssh -p $PORT $HOST << 'EOF'
cd /workspace
source /venv/main/bin/activate
python3 train.py
EOF

# copy results back to local computer
rsync -r --remove-source-files --rsh "ssh -p $PORT" --info=progress2      $HOST:/workspace/exp_* ./ 
# ssh -p $PORT $HOST 'find /home/to/Downloads/ -empty -type d -delete' # 