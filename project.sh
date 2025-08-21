#bin/bash
set -euo pipefail


cd ~/Desktop/ProjectDesignMain
source .venv/bin/activate 
echo $(which python3)

python3 Project-Design-1/main.py
