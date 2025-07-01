# RL Pipeline Architecture: Multi-turn Code Execution Training

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RL TRAINING PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION PHASE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Same model weights loaded into both components:                                │
│                                                                                 │
│  ┌─────────────────────────┐           ┌─────────────────────────┐             │
│  │   Policy Model (TRL)    │◄─────────►│  Completion Server      │             │
│  │   [Training Process]    │           │  [VLLM Inference]       │             │
│  └─────────────────────────┘           └─────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Main Components

### 1. Interactive Code Generation & Evaluation Service
**Components:**
- **Completion Server** (VLLM serving latest model weights)
- **Code Execution Engine** (sandboxed runtime environment)

### 2. Policy Training Module
**Component:**
- **GRPO Trainer** (using TRL framework)

---

## Training Iteration Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING ITERATION                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: Prompt Sampling
┌─────────────────────────┐
│   Policy Training       │
│   Module (TRL)          │
│                         │
│  1) Sample batch of     │
│     prompts from        │
│     training data       │
└─────────────┬───────────┘
              │
              │ 2) Request RL trajectories
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                Interactive Code Generation & Evaluation Service                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────┐         ┌─────────────────────────┐               │
│  │   Completion Server     │         │   Code Execution        │               │
│  │   (VLLM)               │         │   Engine                │               │
│  │                        │         │                         │               │
│  │ 3) Receives prompts    │         │ 6) Executes code        │               │
│  │ 4) Generates multiple  │         │ 7) Returns execution    │               │
│  │    code completions    │         │    results/output       │               │
│  │                        │         │                         │               │
│  └─────────────┬───────────┘         └─────────────┬───────────┘               │
│                │                                   │                           │
│                │ 5) Send code snippets             │                           │
│                └─────────────────────────────────────                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                    Trajectory Management                                    │
│  │                                                                             │
│  │  • Store current trajectory (prompts + code + execution results)           │
│  │  • Concatenate execution output to conversation history                     │
│  │  • Generate next prompt for continuation                                    │
│  │  • Track termination conditions (max steps OR completion criteria)         │
│  │                                                                             │
│  │  8) Repeat steps 4-7 until:                                                │
│  │     - Max steps reached, OR                                                 │
│  │     - Execution completion condition met                                    │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  9) Calculate trajectory rewards based on:                                     │
│     - Successful completion vs max steps termination                           │
│     - Code execution success/failure                                           │
│     - Task-specific scoring metrics                                            │
│                                                                                 │
└─────────────────────────────┬───────────────────────────────────────────────────┘
                              │
                              │ 10) Return training data:
                              │     a) Multi-turn trajectories 
                              │        (code + execution interleaved)
                              │     b) Trajectory-level rewards
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Policy Training Module                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  11) GRPO Training Update:                                                      │
│      • Use multi-turn trajectories as training sequences                       │
│      • Apply trajectory rewards to policy gradient updates                     │
│      • Update model weights                                                     │
│                                                                                 │
│  12) Sync updated weights to Completion Server                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Updated model weights
                              ▼
                    ┌─────────────────────────┐
                    │    Next Iteration       │
                    │   (Repeat 1-12)         │
                    └─────────────────────────┘
```

## Key Data Structures

### Trajectory Format
```
trajectory = {
    "turns": [
        {
            "prompt": "Write a function to calculate fibonacci",
            "code": "def fibonacci(n): ...",
            "execution_output": "Test passed: fibonacci(5) = 5",
            "step": 1
        },
        {
            "prompt": "Now optimize it for large inputs",
            "code": "def fibonacci_optimized(n): ...",
            "execution_output": "Performance improved: 1000x faster",
            "step": 2
        }
    ],
    "final_reward": 0.85,
    "termination_reason": "completion_criteria_met"
}
```

### Reward Calculation
- **Completion bonus**: Higher reward for natural completion vs timeout
- **Execution success**: Bonus for successful code execution
- **Task progress**: Incremental rewards for advancing toward solution
- **Code quality**: Optional scoring for style, efficiency, correctness

---

## Architecture Benefits

1. **Decoupled Training & Inference**: Completion server can be scaled independently
2. **Multi-turn Learning**: Captures interactive coding patterns and debugging flows  
3. **Real Execution Feedback**: Ground truth rewards from actual code execution
4. **Flexible Termination**: Supports both natural completion and bounded exploration
5. **Trajectory-level Optimization**: GRPO trains on full conversation sequences