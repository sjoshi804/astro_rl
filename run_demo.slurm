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
export FITS_FILE_PATH="/work/10450/sjoshi804/vista/astro_rl/astro1_uv_imaging_telescope.fits"

# Service Ports (adjust if needed to avoid conflicts)
export VLLM_SERVER_PORT_1="8200"
export VLLM_SERVER_PORT_2="8201"
export COMPLETION_SERVER_PORT="8000"
export EXEC_ENGINE_PORT="8001"
export CODE_EXEC_SERVICE_PORT="8002"

# Service Configuration
export TENSOR_PARALLEL_SIZE="1"
export MAX_TURNS="8"
export TIMEOUT_SECONDS="60"
export TIMEOUT_PER_STEP="15"
export MAX_STEPS="20"
export PROB_COMPLETION="0.1"

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
EXEC_NODE=${NODES[2]}
ORCHESTRATOR_NODE=${NODES[3]}

# Construct service URLs
export VLLM_SERVER_1_URL="${VLLM_NODE_1}:${VLLM_SERVER_PORT_1}"
export VLLM_SERVER_2_URL="${VLLM_NODE_2}:${VLLM_SERVER_PORT_2}"
export COMPLETION_SERVER_URL="http://${ORCHESTRATOR_NODE}:${COMPLETION_SERVER_PORT}"
export EXEC_ENGINE_URL="http://${EXEC_NODE}:${EXEC_ENGINE_PORT}"
export CODE_EXEC_SERVICE_URL="http://${ORCHESTRATOR_NODE}:${CODE_EXEC_SERVICE_PORT}"

echo "==================================================================="
echo "RL ASTRONOMY PIPELINE JOB STARTING"
echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes allocated: ${NODES[@]}"
echo "VLLM Node 1: $VLLM_NODE_1 (port $VLLM_SERVER_PORT_1)"
echo "VLLM Node 2: $VLLM_NODE_2 (port $VLLM_SERVER_PORT_2)"
echo "Execution Node: $EXEC_NODE (port $EXEC_ENGINE_PORT)"
echo "Orchestrator Node: $ORCHESTRATOR_NODE (ports $COMPLETION_SERVER_PORT, $CODE_EXEC_SERVICE_PORT)"
echo "Model: $MODEL_PATH"
echo "FITS File: $FITS_FILE_PATH"
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
    pkill -f "execution_engine.py" || true
    pkill -f "completion_server.py" || true
    pkill -f "qvice.py" || true
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
# START EXECUTION ENGINE (NODE 2)
# =============================================================================

echo "Starting Execution Engine..."

srun --nodes=1 --ntasks=1 --nodelist=$EXEC_NODE \
    python execution_engine.py \
    --timeout-per-step $TIMEOUT_PER_STEP \
    --max-steps $MAX_STEPS \
    --prob-completion $PROB_COMPLETION \
    --host 0.0.0.0 \
    --port $EXEC_ENGINE_PORT \
    > logs/${SLURM_JOB_ID}/execution_engine.out \
    2> logs/${SLURM_JOB_ID}/execution_engine.err &

EXEC_PID=$!

echo "Execution Engine starting (PID: $EXEC_PID)..."

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

# Wait for execution engine
wait_for_service "$EXEC_ENGINE_URL" "Execution Engine" || exit 1

# Wait for completion server
wait_for_service "$COMPLETION_SERVER_URL" "Completion Server" || exit 1

echo "✓ All core services are ready!"

# =============================================================================
# START CODE GENERATION & EXECUTION SERVICE (NODE 3)
# =============================================================================

echo "Starting Code Generation & Execution Service..."

srun --nodes=1 --ntasks=1 --nodelist=$ORCHESTRATOR_NODE \
    bash -c "cd $WORK/astro_rl && python code_and_exec_service.py \
    --completion-server-url '$COMPLETION_SERVER_URL' \
    --exec-engine-url '$EXEC_ENGINE_URL' \
    --max-turns $MAX_TURNS \
    --timeout $TIMEOUT_SECONDS \
    --trajectory-output-dir '$TRAJECTORY_OUTPUT_DIR' \
    --host 0.0.0.0 \
    --port $CODE_EXEC_SERVICE_PORT" \
    > logs/${SLURM_JOB_ID}/code_exec_service.out \
    2> logs/${SLURM_JOB_ID}/code_exec_service.err &


# Wait for service to be ready
sleep 20
wait_for_service "$CODE_EXEC_SERVICE_URL" "Code Generation & Execution Service" || exit 1

echo "✓ Code Generation & Execution Service is ready!"

# =============================================================================
# RUN DATA GENERATION (NODE 3)
# =============================================================================

echo "Starting Data Generation..."

# Construct data generator arguments
DATA_GEN_ARGS=(
    --service-url "$CODE_EXEC_SERVICE_URL"
    --fits-file "$FITS_FILE_PATH"
    --max-turns $GENERATION_MAX_TURNS
    --timeout $GENERATION_TIMEOUT
    --output-file "astronomy_generation_results_${SLURM_JOB_ID}.json"
)

# Add concurrent execution if enabled
if [ "$GENERATION_CONCURRENT" = "true" ]; then
    DATA_GEN_ARGS+=(--concurrent --concurrency $GENERATION_CONCURRENCY)
fi

# Add code saving if enabled
if [ "$SAVE_CODE_SNIPPETS" = "true" ]; then
    DATA_GEN_ARGS+=(--save-code)
fi

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
    if [ -f "astronomy_generation_results_${SLURM_JOB_ID}.json" ]; then
        echo "Results saved to: astronomy_generation_results_${SLURM_JOB_ID}.json"
    fi
    
    if [ -f "astronomy_code_snippets_*.py" ]; then
        echo "Code snippets saved to: $(ls astronomy_code_snippets_*.py | head -1)"
    fi
    
    # Count generated trajectories
    if [ -d "$TRAJECTORY_OUTPUT_DIR" ]; then
        TRAJECTORY_COUNT=$(find "$TRAJECTORY_OUTPUT_DIR" -name "*.json" | wc -l)
        echo "Generated trajectories: $TRAJECTORY_COUNT"
    fi
    
else
    echo "✗ Data generation failed with exit code: $DATA_GEN_EXIT_CODE"
fi

# Display service status
echo ""
echo "Service Status:"
echo "  VLLM Server 1 (PID $VLLM_PID_1): $(ps -p $VLLM_PID_1 > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  VLLM Server 2 (PID $VLLM_PID_2): $(ps -p $VLLM_PID_2 > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Execution Engine (PID $EXEC_PID): $(ps -p $EXEC_PID > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Completion Server (PID $COMPLETION_PID): $(ps -p $COMPLETION_PID > /dev/null && echo 'Running' || echo 'Stopped')"
echo "  Code Exec Service (PID $CODE_EXEC_PID): $(ps -p $CODE_EXEC_PID > /dev/null && echo 'Running' || echo 'Stopped')"

echo ""
echo "Job completed. Log files:"
echo "  Main SLURM job stdout: astro_rl_data_gen_${SLURM_JOB_ID}.out"
echo "  Main SLURM job stderr: astro_rl_data_gen_${SLURM_JOB_ID}.err"
echo "  Service logs directory: logs/${SLURM_JOB_ID}/"
echo "  VLLM Wrapper Node 1: logs/${SLURM_JOB_ID}/vllm_wrapper_node1.out/.err"
echo "  VLLM Wrapper Node 2: logs/${SLURM_JOB_ID}/vllm_wrapper_node2.out/.err"
echo "  Execution Engine: logs/${SLURM_JOB_ID}/execution_engine.out/.err"
echo "  Completion Server: logs/${SLURM_JOB_ID}/completion_server.out/.err"
echo "  Code Exec Service: logs/${SLURM_JOB_ID}/code_exec_service.out/.err"
echo "  Data Generator: logs/${SLURM_JOB_ID}/data_generator.out/.err"
echo "==================================================================="

# Exit with data generation exit code
exit $DATA_GEN_EXIT_CODE