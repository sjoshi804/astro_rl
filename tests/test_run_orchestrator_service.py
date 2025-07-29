"""
Unit tests for run_orchestrator_service.py
Tests the orchestrator service creation and configuration with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from fastapi import FastAPI

# Import the module under test
from multi_turn_rl.scripts.run_orchestrator_service import (
    parse_args,
    create_orchestrator_service,
    main
)


class TestParseArgs:
    """Test argument parsing functionality"""
    
    def test_parse_args_with_required_params(self):
        """Test parsing with minimum required arguments"""
        test_args = [
            "--run-id", "test123",
            "--vllm-servers", "http://localhost:8000", "http://localhost:8001"
        ]
        
        with patch('sys.argv', ['script'] + test_args):
            args = parse_args()
            
        assert args.run_id == "test123"
        assert args.vllm_servers == ["http://localhost:8000", "http://localhost:8001"]
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.max_turns == 10
    
    def test_parse_args_with_all_params(self):
        """Test parsing with all arguments specified"""
        test_args = [
            "--run-id", "test456",
            "--vllm-servers", "http://server1:8000",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--max-turns", "15",
            "--timeout-seconds", "45",
            "--ray-timeout-per-step", "60.0",
            "--ray-num-cpus", "2",
            "--ray-num-gpus", "1",
            "--load-balancer-strategy", "random",
            "--health-check-interval", "60",
            "--max-retries", "5"
        ]
        
        with patch('sys.argv', ['script'] + test_args):
            args = parse_args()
            
        assert args.run_id == "test456"
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.max_turns == 15
        assert args.timeout_seconds == 45
        assert args.ray_timeout_per_step == 60.0
        assert args.ray_num_cpus == 2
        assert args.ray_num_gpus == 1
        assert args.load_balancer_strategy == "random"
        assert args.health_check_interval == 60
        assert args.max_retries == 5


class TestCreateOrchestratorService:
    """Test orchestrator service creation with mocked dependencies"""
    
    @patch('multi_turn_rl.scripts.run_orchestrator_service.create_app')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.Orchestrator')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.CompletionLoadBalancer')
    @patch('pathlib.Path.mkdir')
    def test_create_orchestrator_service_basic(self, mock_mkdir, mock_lb_class, mock_orch_class, mock_create_app):
        """Test basic service creation with mocked dependencies"""
        # Setup mocks
        mock_lb_instance = Mock()
        mock_orch_instance = Mock()
        mock_app = Mock(spec=FastAPI)
        
        mock_lb_class.return_value = mock_lb_instance
        mock_orch_class.return_value = mock_orch_instance
        mock_create_app.return_value = mock_app
        
        # Create mock args
        mock_args = Mock()
        mock_args.run_id = "test123"
        mock_args.vllm_servers = ["http://localhost:8000"]
        mock_args.load_balancer_strategy = "round_robin"
        mock_args.health_check_interval = 30
        mock_args.max_retries = 3
        mock_args.max_turns = 10
        mock_args.timeout_seconds = 30
        mock_args.ray_timeout_per_step = 30.0
        mock_args.ray_num_cpus = 1
        mock_args.ray_num_gpus = 0
        
        # Call the function
        result = create_orchestrator_service(mock_args)
        
        # Verify directory creation
        assert mock_mkdir.call_count == 2  # trajectories and logs dirs
        
        # Verify CompletionLoadBalancer creation
        mock_lb_class.assert_called_once_with(
            vllm_servers=["http://localhost:8000"],
            strategy="round_robin",
            health_check_interval=30,
            max_retries=3
        )
        
        # Verify Orchestrator creation
        mock_orch_class.assert_called_once()
        call_args = mock_orch_class.call_args[1]
        assert call_args['completion_load_balancer'] == mock_lb_instance
        assert call_args['max_turns'] == 10
        assert call_args['timeout_seconds'] == 30
        assert call_args['ray_timeout_per_step'] == 30.0
        assert call_args['ray_num_cpus'] == 1
        assert call_args['ray_num_gpus'] == 0
        assert 'trajectories/test123' in call_args['trajectory_output_dir']
        assert 'trajectories/test123/prompts.jsonl' in call_args['prompts_jsonl_path']
        
        # Verify create_app call
        mock_create_app.assert_called_once_with(mock_orch_instance, mock_lb_instance)
        
        # Verify return value
        assert result == mock_app
    
    @patch('multi_turn_rl.scripts.run_orchestrator_service.create_app')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.Orchestrator')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.CompletionLoadBalancer')
    @patch('pathlib.Path.mkdir')
    def test_create_orchestrator_service_custom_config(self, mock_mkdir, mock_lb_class, mock_orch_class, mock_create_app):
        """Test service creation with custom configuration"""
        # Setup mocks
        mock_lb_instance = Mock()
        mock_orch_instance = Mock()
        mock_app = Mock(spec=FastAPI)
        
        mock_lb_class.return_value = mock_lb_instance
        mock_orch_class.return_value = mock_orch_instance
        mock_create_app.return_value = mock_app
        
        # Create mock args with custom values
        mock_args = Mock()
        mock_args.run_id = "custom456"
        mock_args.vllm_servers = ["http://server1:8000", "http://server2:8000"]
        mock_args.load_balancer_strategy = "random"
        mock_args.health_check_interval = 60
        mock_args.max_retries = 5
        mock_args.max_turns = 20
        mock_args.timeout_seconds = 60
        mock_args.ray_timeout_per_step = 45.0
        mock_args.ray_num_cpus = 4
        mock_args.ray_num_gpus = 2
        
        # Call the function
        result = create_orchestrator_service(mock_args)
        
        # Verify CompletionLoadBalancer creation with custom values
        mock_lb_class.assert_called_once_with(
            vllm_servers=["http://server1:8000", "http://server2:8000"],
            strategy="random",
            health_check_interval=60,
            max_retries=5
        )
        
        # Verify Orchestrator creation with custom values
        call_args = mock_orch_class.call_args[1]
        assert call_args['max_turns'] == 20
        assert call_args['timeout_seconds'] == 60
        assert call_args['ray_timeout_per_step'] == 45.0
        assert call_args['ray_num_cpus'] == 4
        assert call_args['ray_num_gpus'] == 2


class TestMainFunction:
    """Test the main function with mocked dependencies"""
    
    @patch('multi_turn_rl.scripts.run_orchestrator_service.uvicorn.run')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.create_orchestrator_service')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.parse_args')
    def test_main_function_success(self, mock_parse_args, mock_create_service, mock_uvicorn_run):
        """Test successful execution of main function"""
        # Setup mocks
        mock_args = Mock()
        mock_args.host = "0.0.0.0"
        mock_args.port = 8080
        mock_args.run_id = "test789"
        mock_args.vllm_servers = ["http://localhost:8000"]
        mock_args.max_turns = 10
        mock_args.ray_num_cpus = 1
        mock_args.ray_num_gpus = 0
        
        mock_app = Mock(spec=FastAPI)
        mock_parse_args.return_value = mock_args
        mock_create_service.return_value = mock_app
        
        # Call main function
        main()
        
        # Verify function calls
        mock_parse_args.assert_called_once()
        mock_create_service.assert_called_once_with(mock_args)
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
    
    @patch('multi_turn_rl.scripts.run_orchestrator_service.uvicorn.run')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.create_orchestrator_service')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.parse_args')
    @patch('sys.exit')
    def test_main_function_exception_handling(self, mock_exit, mock_parse_args, mock_create_service, mock_uvicorn_run):
        """Test main function exception handling in uvicorn.run"""
        # Setup mocks
        mock_args = Mock()
        mock_args.host = "0.0.0.0"
        mock_args.port = 8080
        mock_app = Mock(spec=FastAPI)
        
        mock_parse_args.return_value = mock_args
        mock_create_service.return_value = mock_app
        mock_uvicorn_run.side_effect = Exception("Test exception")
        
        # Call main function
        main()
        
        # Verify sys.exit was called with error code
        mock_exit.assert_called_once_with(1)
    
    @patch('multi_turn_rl.scripts.run_orchestrator_service.uvicorn.run')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.create_orchestrator_service')
    @patch('multi_turn_rl.scripts.run_orchestrator_service.parse_args')
    def test_main_function_keyboard_interrupt(self, mock_parse_args, mock_create_service, mock_uvicorn_run):
        """Test main function handles KeyboardInterrupt gracefully"""
        # Setup mocks
        mock_args = Mock()
        mock_app = Mock(spec=FastAPI)
        mock_parse_args.return_value = mock_args
        mock_create_service.return_value = mock_app
        mock_uvicorn_run.side_effect = KeyboardInterrupt()
        
        # Call main function - should not raise exception
        main()
        
        # Verify that uvicorn.run was called
        mock_uvicorn_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])