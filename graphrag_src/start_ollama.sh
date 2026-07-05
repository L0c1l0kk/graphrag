#!/bin/bash
# start_ollama.sh
export PATH=$HOME/.local/ollama/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/ollama/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_CONTEXT_LENGTH=4096

nohup ollama serve > $HOME/ollama.log 2>&1 &
disown
echo "Ollama started, PID $!, logs at ~/ollama.log"