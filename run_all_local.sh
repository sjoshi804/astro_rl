#!/bin/bash

set -e

# Start Ray cluster (head node)
echo "Starting Ray head node..."
ray stop --force || true
ray start --head --port=6379 &

# Start mock completion server
echo "Starting mock completion server on port 8000..."
python mock_completion_server.py > mock_completion_server.log 2>&1 &

# Start code execution service (if you have one)
if [ -f code_and_exec_service.py ]; then
    echo "Starting code execution service on port 8002..."
    python code_and_exec_service.py --host 0.0.0.0 --port 8002 > code_exec_service.log 2>&1 &
else
    echo "No code_and_exec_service.py found, skipping code execution service."
fi

# Start VLLM wrapper (if you want to test with real VLLM)
if [ -f vllm_wrapper.py ]; then
    echo "Starting VLLM wrapper on port 8200..."
    python vllm_wrapper.py --model-path mock-model --port 8200 --host 0.0.0.0 > vllm_wrapper.log 2>&1 &
else
    echo "No vllm_wrapper.py found, skipping VLLM wrapper."
fi

echo "All services started. Check logs for output."
echo "To stop all, run: pkill -f 'python mock_completion_server.py'; pkill -f 'python code_and_exec_service.py'; pkill -f 'python vllm_wrapper.py'; ray stop --force"