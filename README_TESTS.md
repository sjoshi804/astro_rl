# Debug Tests for VLLM Pipeline Issues

This directory contains comprehensive unit tests to debug the VLLM pipeline issues we've been experiencing.

## Quick Start

### Run Manual Tests (Quick Debugging)
```bash
python manual_test.py
```

### Run Full Test Suite
```bash
python test_debug.py
```

## What the Tests Cover

### 1. Code Extraction & Validation (Offline)
- ✅ Tests if `extract_python_code()` properly extracts Python from markdown
- ✅ Tests if natural language gets rejected and falls back to safe code
- ✅ Tests if `is_valid_python_code()` correctly validates Python syntax

### 2. Prompt Building (Offline)
- ✅ Tests if our improved prompt structure includes all necessary elements
- ✅ Verifies prompts encourage code-only generation
- ✅ Tests both empty trajectory and history-based prompts

### 3. Ray Execution Engine (Requires Ray Cluster)
- ✅ Tests Ray instance creation and cleanup
- ✅ Tests execution of valid Python code
- ✅ Tests handling of invalid Python (natural language)
- ✅ Verifies "crashed" state is returned, not actual actor crashes

### 4. VLLM Wrapper (Requires VLLM Service)
- ✅ Tests VLLM health and connectivity
- ✅ Tests response to empty prompts (potential cause of empty outputs)
- ✅ Tests response to our improved prompt structure
- ✅ Validates parameter schema compatibility

### 5. Completion Server (Requires Completion Service)
- ✅ Tests completion server health and connectivity
- ✅ Tests generation with real request structures
- ✅ Tests the full prompt building → VLLM → response flow

### 6. Full Pipeline (Requires All Services)
- ✅ End-to-end test of the complete trajectory generation
- ✅ Tests if final output contains valid Python code
- ✅ Identifies where in the pipeline failures occur

## Expected Service URLs

The tests assume services are running on:
- **VLLM Wrapper 1**: `http://localhost:8200`
- **VLLM Wrapper 2**: `http://localhost:8201` 
- **Completion Server**: `http://localhost:8000`
- **Code Execution Service**: `http://localhost:8002`
- **Ray Cluster**: Local Ray cluster

## Interpreting Results

### ✅ Green Tests = Working
- Component is functioning correctly
- No issues found in this part of the pipeline

### ❌ Red Tests = Issues Found
- Component has problems that need fixing
- Check the error details for specific failure reasons

## Common Issues and Fixes

### "VLLM generated empty outputs list"
- **Cause**: Parameter schema mismatch or empty prompts
- **Test**: VLLM wrapper direct tests
- **Fix**: Update request models in `vllm_wrapper.py`

### "This Python code accomplishes the task by:"
- **Cause**: Poor prompt structure encouraging natural language
- **Test**: Prompt building and completion server tests
- **Fix**: Improve prompts in `completion_server.py`

### "SyntaxError: invalid syntax" 
- **Cause**: Natural language being executed as Python
- **Test**: Code extraction and Ray execution tests
- **Fix**: Improve code extraction in `code_and_exec_service.py`

### Ray actors "crashing"
- **Test**: Ray execution tests
- **Note**: "crashed" is just a state, not actual actor failure

## Running Tests with Services Down

Some tests can run offline:
- Code Extraction & Validation ✅
- Prompt Building ✅ 
- Ray Execution (if Ray cluster is running) ✅

Service-dependent tests will fail gracefully and report connection issues. 