#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/config.sh

# # 1. install pipx 2. install tldr 3. config .bashrc
ssh -p $PORT $HOST << 'EOF'
sudo apt update
sudo apt install pipx -y
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

pipx install tldr
tldr --update

echo "
stty -ixon
shopt -s globstar
alias ls='ls -XFh --color=auto'
alias ll='ls -l'
alias la='ls -A'
alias lla='ls -lA'
alias nemo='nemo --existing-window'" >> ~/.bashrc
source ~/.bashrc
EOF

# copy dataset and python files to server
rsync --rsh "ssh -p $PORT" --info=progress2           $LOCAL_ROOT/*.py $HOST:$REMOTE_ROOT
rsync -r --rsh "ssh -p $PORT" --info=progress2        $LOCAL_ROOT/BB_Dataset_* $HOST:$REMOTE_ROOT

# install requirements
rsync --rsh "ssh -p $PORT" --info=progress2           $LOCAL_ROOT/requirements.txt  $HOST:$REMOTE_ROOT
ssh -p $PORT $HOST "
/venv/main/bin/pip install -r $REMOTE_ROOT/requirements.txt
rm $REMOTE_ROOT/requirements.txt
"
