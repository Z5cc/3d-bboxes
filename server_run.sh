#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/config.sh

# copy python scripts to server
rsync --rsh "ssh -p $PORT" --info=progress2      $LOCAL_ROOT/*.py $HOST:$REMOTE_ROOT

# train
ssh -p $PORT $HOST << 'EOF'
cd /workspace
source /venv/main/bin/activate
python3 train.py
EOF

# copy results back to local computer
rsync -r --remove-source-files --rsh "ssh -p $PORT" --info=progress2      $HOST:$REMOTE_ROOT/exp_* $LOCAL_ROOT
ssh -p $PORT $HOST 'find /workspace/ -type d -empty -delete'
