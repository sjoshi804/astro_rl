# Multi-Turn RL

A reinforcement learning framework for multi-turn code generation and execution.

## Overview

Multi-Turn RL provides a comprehensive framework for:
- **Multi-turn code generation** using language models
- **Isolated code execution** with Ray actors
- **Load balancing** across multiple VLLM servers
- **Trajectory tracking** for RL training

## Key Components

- **Orchestrator** - Manages trajectory generation and execution
- **CompletionLoadBalancer** - Distributes requests across VLLM servers  
- **RayExecutionEngine** - Handles isolated code execution
- **Models** - Pydantic data models for trajectories and requests

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from multi_turn_rl import Orchestrator, CompletionLoadBalancer
from multi_turn_rl.models import BatchTrajectoryRequest, TrajectoryRequest

# Create load balancer
load_balancer = CompletionLoadBalancer(
    vllm_hostnames=["localhost:8000", "localhost:8001"]
)

# Create orchestrator
orchestrator = Orchestrator(
    completion_load_balancer=load_balancer,
    max_turns=10,
    trajectory_output_dir="./trajectories"
)

# Generate trajectories
request = BatchTrajectoryRequest(
    requests=[
        TrajectoryRequest(
            initial_prompts=["Write a function to calculate fibonacci numbers"],
            max_turns=5
        )
    ]
)

trajectories = await orchestrator.generate_trajectories(request)
```

### Running the Service

```python
from multi_turn_rl.main import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8002)
```

## Architecture

The framework uses direct function calls between components instead of HTTP:

```
Client Code → Orchestrator → CompletionLoadBalancer → VLLM Servers
                ↓
            RayExecutionEngine
```

## Features

- **Trajectory Isolation** - Each trajectory gets its own Ray actor
- **Proper Ray Management** - Automatic cleanup and resource management
- **Load Balancing** - Round-robin distribution across VLLM servers
- **JSONL Logging** - All prompts logged for analysis
- **Success Criteria** - Configurable task completion detection

## Requirements

- Python 3.8+
- Ray 2.0+
- FastAPI
- HTTPX
- Pydantic