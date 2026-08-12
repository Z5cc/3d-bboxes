#!/usr/bin/env bash

source ./config.sh

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
rsync --rsh "ssh -p $PORT" --info=progress2 *.py $HOST:/workspace/
rsync -r --rsh "ssh -p $PORT" --info=progress2 dl_challenge_* $HOST:/workspace/

# install requirements
rsync --rsh "ssh -p $PORT" --info=progress2 requirements.txt $HOST:/workspace/
ssh -p $PORT $HOST '/venv/main/bin/pip install -r /workspace/requirements.txt && rm /workspace/requirements.txt'
