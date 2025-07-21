#!/bin/bash

###################
# SLURM Configuration 
###################
#SBATCH --job-name=astro_rl_data_gen        # Job name 
#SBATCH --nodes=5                  # Number of nodes
#SBATCH --ntasks-per-node=4       
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00           # Max job run time (HH:MM:SS)
#SBATCH --output=/work/10450/sjoshi804/vista/astro_rl/logs/astro_rl_data_gen_%j.out
#SBATCH --error=/work/10450/sjoshi804/vista/astro_rl/logs/astro_rl_data_gen_%j.err   # Error log
#SBATCH --partition=gh            # Partition name
#SBATCH --export=ALL

###################
# DEMO CONFIGURATION GUIDE
###################
# To run different demo types, change the DEMO_CONFIG variable below:
#
# Available Configurations:
# - "quick"              : Fast 2-task demo (~3 min) - fibonacci, data analysis
# - "standard"           : Balanced 4-task demo (~8 min) - mixed categories  
# - "comprehensive"      : Full 8-task demo (~15 min) - all categories
# - "astronomy_focused"  : FITS analysis tasks (~10 min) - astronomy only
# - "programming_focused": Programming tasks (~10 min) - coding challenges
# - "output_tests"       : Output capture tests (~5 min) - verify stdout/stderr capture
# - "multiturn_tests"    : Multi-turn interaction tests (~8 min) - verify multi-step functionality
#
# Simply change DEMO_CONFIG="standard" to your desired configuration below!
###################


# =============================================================================
# ENVIRONMENT VARIABLES - CONFIGURE ALL PARAMETERS HERE
# =============================================================================


# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate vllm
cd $WORK/astro_rl

# Load modules
echo "Loading modules..."
module load gcc/15 
module load gcc/15 cuda/12.8 nccl/12.4
module list

# Model and Data Configuration
export MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
export FITS_FILE_PATH="/work/10450/sjoshi804/vista/astro_rl/demo_data/astro1_uv_imaging_telescope.fits"

# Service Ports (adjust if needed to avoid conflicts)
export VLLM_SERVER_PORT_1="8200"
export VLLM_SERVER_PORT_2="8201"
export COMPLETION_SERVER_PORT="8000"
export CODE_EXEC_SERVICE_PORT="8002"

# Service Configuration
export TENSOR_PARALLEL_SIZE="1"
export MAX_TURNS="10"
export TIMEOUT_SECONDS="120"
export TIMEOUT_PER_STEP="120"
export MAX_STEPS="10"
export PROB_COMPLETION="0.1"
export CONDA_ENV="astro_rl"

# =============================================================================
# DEMO CONFIGURATION - CHANGE THIS TO SELECT DEMO TYPE
# =============================================================================
# Available configs: quick, standard, comprehensive, astronomy_focused, programming_focused, output_tests, multiturn_tests
export DEMO_CONFIG="output_tests"  # <-- CHANGE THIS LINE TO SELECT DEMO TYPE

# Data Generation Configuration
export TRAJECTORY_OUTPUT_DIR="./trajectories"
export GENERATION_TIMEOUT="180"
export GENERATION_MAX_TURNS="8"
export GENERATION_CONCURRENT="true"
export GENERATION_CONCURRENCY="3"
export SAVE_CODE_SNIPPETS="true"

# Health Check Configuration
export HEALTH_CHECK_INTERVAL="10"
export REQUEST_TIMEOUT="60"
export STARTUP_WAIT_TIME="30"

# Logging
export LOG_LEVEL="INFO"

# =============================================================================
# DERIVED VARIABLES (DO NOT MODIFY)
# =============================================================================

# Get node hostnames
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
VLLM_NODE_1=${NODES[0]}
VLLM_NODE_2=${NODES[1]}
ORCHESTRATOR_NODE=${NODES[2]}

# Construct service URLs
export VLLM_SERVER_1_URL="${VLLM_NODE_1}:${VLLM_SERVER_PORT_1}"
export VLLM_SERVER_2_URL="${VLLM_NODE_2}:${VLLM_SERVER_PORT_2}"
export COMPLETION_SERVER_URL="http://${ORCHESTRATOR_NODE}:${COMPLETION_SERVER_PORT}"
export CODE_EXEC_SERVICE_URL="http://${ORCHESTRATOR_NODE}:${CODE_EXEC_SERVICE_PORT}"

echo "==================================================================="
echo "RL ASTRONOMY PIPELINE JOB STARTING"
echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes allocated: ${NODES[@]}"
echo "VLLM Node 1: $VLLM_NODE_1 (port $VLLM_SERVER_PORT_1)"
echo "VLLM Node 2: $VLLM_NODE_2 (port $VLLM_SERVER_PORT_2)"
echo "Orchestrator Node: $ORCHESTRATOR_NODE (ports $COMPLETION_SERVER_PORT, $CODE_EXEC_SERVICE_PORT)"
echo "Ray Cluster: All ${SLURM_JOB_NUM_NODES} nodes (head: ${NODES[0]})"
echo "Model: $MODEL_PATH"
echo "FITS File: $FITS_FILE_PATH"
echo "Demo Configuration: $DEMO_CONFIG"
echo "Conda Environment: $CONDA_ENV"
echo "==================================================================="

# Create necessary directories
mkdir -p logs
mkdir -p logs/${SLURM_JOB_ID}
mkdir -p $TRAJECTORY_OUTPUT_DIR

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=120
    local attempt=1
    
    echo "Waiting for $service_name at $url..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo "✓ $service_name is ready!"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 5
        ((attempt++))
    done
    
    echo "✗ $service_name failed to start after $((max_attempts * 5)) seconds"
    return 1
}

cleanup_services() {
    echo "Cleaning up services..."
    pkill -f "vllm_wrapper.py" || true
    pkill -f "completion_server.py" || true
    pkill -f "code_and_exec_service.py" || true
    
    # Clean up Ray cluster
    echo "Stopping Ray cluster..."
    for node in "${NODES[@]}"; do
        echo "Stopping Ray on node: $node"
        srun --nodes=1 --ntasks=1 -w "$node" ray stop --force > /dev/null 2>&1 || true
    done
    
    sleep 5
}

# Set up cleanup trap
trap cleanup_services EXIT

# =============================================================================
# START VLLM SERVERS (NODES 0 AND 1)
# =============================================================================

echo "Starting VLLM servers..."

# Start VLLM Wrapper on Node 1
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE_1 \
    python vllm_wrapper.py \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $VLLM_SERVER_PORT_1 \
    --host 0.0.0.0 \
    > logs/${SLURM_JOB_ID}/vllm_wrapper_node1.out \
    2> logs/${SLURM_JOB_ID}/vllm_wrapper_node1.err &

VLLM_PID_1=$!

# Start VLLM Wrapper on Node 2
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE_2 \
    python vllm_wrapper.py \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $VLLM_SERVER_PORT_2 \
    --host 0.0.0.0 \
    > logs/${SLURM_JOB_ID}/vllm_wrapper_node2.out \
    2> logs/${SLURM_JOB_ID}/vllm_wrapper_node2.err &

VLLM_PID_2=$!

echo "VLLM servers starting (PIDs: $VLLM_PID_1, $VLLM_PID_2)..."

# =============================================================================
# START RAY CLUSTER (ALL NODES)
# =============================================================================

echo "Setting up Ray cluster..."

# Determine GPUs per node (if SLURM provides this info; default to 1)
gpus_per_node=${SLURM_GPUS_PER_NODE:-1}

# Get the head node (first node)
head_node=${NODES[0]}
echo "Ray head node: $head_node"

# Get head node IP
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle potential IPv6 addresses
if [[ "$head_node_ip" == *" "* ]]; then  
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "Detected multiple IPs, using: $head_node_ip"
fi

ray_port=6379
ip_head=$head_node_ip:$ray_port
export ip_head
echo "Ray Head Node Address: $ip_head"

# Cleanup previous Ray instances on all nodes
echo "Cleaning up previous Ray instances..."
for node in "${NODES[@]}"; do
    echo "Stopping Ray on node: $node"
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force > /dev/null 2>&1 || true
    srun --nodes=1 --ntasks=1 -w "$node" rm -rf /tmp/ray/* > /dev/null 2>&1 || true
done
sleep 2

# Start the head node
echo "Starting Ray head node on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$ray_port \
    --num-cpus=72 --num-gpus=$gpus_per_node --block \
    > logs/${SLURM_JOB_ID}/ray_head.out \
    2> logs/${SLURM_JOB_ID}/ray_head.err &

RAY_HEAD_PID=$!
sleep 10  # Give head node time to initialize

# Start worker nodes (if we have more than 1 node)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
RAY_WORKER_PIDS=()

if [ $worker_num -gt 0 ]; then
    echo "Starting $worker_num Ray worker nodes..."
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${NODES[$i]}
        echo "Starting Ray worker on node: $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus=72 --num-gpus=$gpus_per_node --block \
            > logs/${SLURM_JOB_ID}/ray_worker_$i.out \
            2> logs/${SLURM_JOB_ID}/ray_worker_$i.err &
        
        RAY_WORKER_PIDS+=($!)
        sleep 5
    done
fi

echo "Waiting for Ray cluster to stabilize..."
sleep 10

# Check Ray cluster status
echo "Ray cluster status:"
srun --nodes=1 --ntasks=1 -w "$head_node" ray status || echo "Ray status check failed"

echo "Ray cluster setup complete!"

# =============================================================================
# START COMPLETION SERVER (NODE 3)
# =============================================================================

echo "Starting Completion Server..."

srun --nodes=1 --ntasks=1 --nodelist=$ORCHESTRATOR_NODE \
    python completion_server.py \
    --vllm-servers "$VLLM_SERVER_1_URL" "$VLLM_SERVER_2_URL" \
    --host 0.0.0.0 \
    --port $COMPLETION_SERVER_PORT \
    --health-check-interval $HEALTH_CHECK_INTERVAL \
    --request-timeout $REQUEST_TIMEOUT \
    > logs/${SLURM_JOB_ID}/completion_server.out \
    2> logs/${SLURM_JOB_ID}/completion_server.err &

COMPLETION_PID=$!

echo "Completion Server starting (PID: $COMPLETION_PID)..."

# =============================================================================
# WAIT FOR CORE SERVICES TO BE READY
# =============================================================================

echo "Waiting for core services to start up..."
sleep $STARTUP_WAIT_TIME

# Wait for VLLM servers
wait_for_service "http://$VLLM_SERVER_1_URL" "VLLM Server 1" || exit 1
wait_for_service "http://$VLLM_SERVER_2_URL" "VLLM Server 2" || exit 1

# Wait for completion server
wait_for_service "$COMPLETION_SERVER_URL" "Completion Server" || exit 1

echo "✓ All core services are ready!"
echo "✓ Ray cluster is running with ${SLURM_JOB_NUM_NODES} nodes"

# =============================================================================
# START CODE GENERATION & EXECUTION SERVICE (NODE 3)
# =============================================================================

echo "Starting Code Generation & Execution Service..."

srun --nodes=1 --ntasks=1 --nodelist=$ORCHESTRATOR_NODE \
    bash -c "cd $WORK/astro_rl && python code_and_exec_service.py \
    --completion-server-url '$COMPLETION_SERVER_URL' \
    --max-turns $MAX_TURNS \
    --timeout $TIMEOUT_SECONDS \
    --ray-timeout-per-step $TIMEOUT_PER_STEP \
    --ray-num-cpus 2 \
    --ray-num-gpus 0 \
    --trajectory-output-dir '$TRAJECTORY_OUTPUT_DIR' \
    --host 0.0.0.0 \
    --port $CODE_EXEC_SERVICE_PORT" \
    > logs/${SLURM_JOB_ID}/code_exec_service.out \
    2> logs/${SLURM_JOB_ID}/code_exec_service.err &

CODE_EXEC_PID=$!


# Wait for service to be ready
sleep 20
if ! wait_for_service "$CODE_EXEC_SERVICE_URL" "Code Generation & Execution Service"; then
    echo "✗ CRITICAL: Code Generation & Execution Service failed to start!"
    echo "✗ Cannot proceed with data generation - all trajectories would be fake failures."
    echo "✗ Check logs/${SLURM_JOB_ID}/code_exec_service.err for details."
    exit 1
fi

echo "✓ Code Generation & Execution Service is ready!"

# =============================================================================
# RUN DATA GENERATION (NODE 3)
# =============================================================================

echo "Starting Data Generation..."

# Construct data generator arguments using demo configuration
DATA_GEN_ARGS=(
    --service-url "$CODE_EXEC_SERVICE_URL"
    --fits-file "$FITS_FILE_PATH"
    --config "$DEMO_CONFIG"
    --output-file "demo_results_${DEMO_CONFIG}_${SLURM_JOB_ID}.json"
    --trajectory-dir "$TRAJECTORY_OUTPUT_DIR"
)

# Add code saving if enabled
if [ "$SAVE_CODE_SNIPPETS" = "true" ]; then
    DATA_GEN_ARGS+=(--save-code)
fi

echo "Running demo with configuration: $DEMO_CONFIG"
echo "Data generator arguments: ${DATA_GEN_ARGS[@]}"

# Run data generation
srun --nodes=1 --ntasks=1 --nodelist=$ORCHESTRATOR_NODE \
    python astro_data_generator.py "${DATA_GEN_ARGS[@]}" \
    > logs/${SLURM_JOB_ID}/data_generator.out \
    2> logs/${SLURM_JOB_ID}/data_generator.err

DATA_GEN_EXIT_CODE=$?

# =============================================================================
# RESULTS AND CLEANUP
# =============================================================================

echo "==================================================================="
echo "DATA GENERATION COMPLETED"
echo "==================================================================="

if [ $DATA_GEN_EXIT_CODE -eq 0 ]; then
    echo "✓ Data generation completed successfully!"
    
    # Display results summary
    RESULTS_FILE="$TRAJECTORY_OUTPUT_DIR/demo_results_${DEMO_CONFIG}_${SLURM_JOB_ID}.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Results saved to: $RESULTS_FILE"
        echo "Demo configuration used: $DEMO_CONFIG"
    fi
    
    # Check for code snippets in trajectory directory
    CODE_SNIPPETS_FILE=$(find "$TRAJECTORY_OUTPUT_DIR" -name "*code_snippets_*.py" -type f | head -1)
    if [ -n "$CODE_SNIPPETS_FILE" ]; then
        echo "Code snippets saved to: $CODE_SNIPPETS_FILE"
    fi
    
    # Count generated trajectory files
    if [ -d "$TRAJECTORY_OUTPUT_DIR" ]; then
        TRAJECTORY_COUNT=$(find "$TRAJECTORY_OUTPUT_DIR" -name "trajectory_*.json" | wc -l)
        TOTAL_FILES=$(find "$TRAJECTORY_OUTPUT_DIR" -type f | wc -l)
        echo "Generated trajectory files: $TRAJECTORY_COUNT"
        echo "Total files in trajectory directory: $TOTAL_FILES"
    fi
    
else
    echo "✗ Data generation failed with exit code: $DATA_GEN_EXIT_CODE"
fi

# Display service status
echo ""
echo "Service Status:"
echo "  VLLM Server 1 (PID $VLLM_PID_1): $(ps -p $VLLM_PID_1 > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  VLLM Server 2 (PID $VLLM_PID_2): $(ps -p $VLLM_PID_2 > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Ray Head Node (PID $RAY_HEAD_PID): $(ps -p $RAY_HEAD_PID > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Ray Worker Nodes: ${#RAY_WORKER_PIDS[@]} workers"
echo "  Completion Server (PID $COMPLETION_PID): $(ps -p $COMPLETION_PID > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Code Exec Service (PID $CODE_EXEC_PID): $(ps -p $CODE_EXEC_PID > /dev/null && echo 'Running' || echo 'Stopped')"

echo ""
echo "Job completed. Log files:"
echo "  Main SLURM job stdout: astro_rl_data_gen_${SLURM_JOB_ID}.out"
echo "  Main SLURM job stderr: astro_rl_data_gen_${SLURM_JOB_ID}.err"
echo "  Service logs directory: logs/${SLURM_JOB_ID}/"
echo "  VLLM Wrapper Node 1: logs/${SLURM_JOB_ID}/vllm_wrapper_node1.out/.err"
echo "  VLLM Wrapper Node 2: logs/${SLURM_JOB_ID}/vllm_wrapper_node2.out/.err"
echo "  Ray Head Node: logs/${SLURM_JOB_ID}/ray_head.out/.err"
echo "  Ray Worker Nodes: logs/${SLURM_JOB_ID}/ray_worker_*.out/.err"
echo "  Completion Server: logs/${SLURM_JOB_ID}/completion_server.out/.err"
echo "  Code Exec Service: logs/${SLURM_JOB_ID}/code_exec_service.out/.err"
echo "  Data Generator: logs/${SLURM_JOB_ID}/data_generator.out/.err"
echo "==================================================================="

# Exit with data generation exit code
exit $DATA_GEN_EXIT_CODE