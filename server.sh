#!/usr/bin/env bash

server_port=40047
server_address=root@83.60.44.26

scp -P $server_port *.py $server_address:/workspace/
ssh -p $server_port $server_address -L 8080:localhost:8080 \
'cd /workspace && source /venv/main/bin/activate && python3 train.py'
# scp -P 40047 
