#!/bin/bash
# start_ollama.sh
export PATH=$HOME/.local/ollama/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/ollama/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_CONTEXT_LENGTH=4048
export LLAMA_ARG_CACHE_RAM=0

nohup ollama serve > $HOME/ollama.log 2>&1 &
disown

ollama create forced-gpu-model -f ./modelfile
echo "Ollama started, PID $!, logs at ~/ollama.log"