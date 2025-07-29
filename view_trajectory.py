#!/usr/bin/env python3
"""
View Trajectory Tool
Formats trajectory JSON files in the chat template format with raw model output
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_trajectory(file_path: str) -> Dict[str, Any]:
    """Load trajectory from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Trajectory file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in trajectory file: {e}")
        sys.exit(1)


def format_trajectory_chat(trajectory: Dict[str, Any]) -> str:
    """Format trajectory in chat template format with raw model output"""
    
    # Get the initial goal from the first turn's prompt
    initial_goal = ""
    if trajectory.get("turns") and len(trajectory["turns"]) > 0:
        initial_goal = trajectory["turns"][0].get("prompt", "")
    
    # Start with the goal
    formatted = f"<goal>\n{initial_goal}\n</goal>\n\n"
    
    # Process each turn
    for turn in trajectory.get("turns", []):
        # Add raw model output if available
        raw_output = turn.get("raw_model_output")
        if raw_output:
            formatted += f"<raw_model_output>\n{raw_output}\n</raw_model_output>\n\n"
        
        # Add the turn with parsed code
        code = turn.get("code", "")
        formatted += f"<turn>\n{code}\n</turn>\n"
        
        # Add execution output if present
        execution_output = turn.get("execution_output", "")
        if execution_output and execution_output.strip():
            formatted += f"<output>\n{execution_output}\n</output>\n"
        
        # Add error output if execution failed
        if not turn.get("execution_success", True):
            error_msg = turn.get("execution_output", "Execution failed")
            formatted += f"<error>\n{error_msg}\n</error>\n"
        
        formatted += "\n"
    
    return formatted


def print_trajectory_info(trajectory: Dict[str, Any]):
    """Print basic information about the trajectory"""
    print("=" * 80)
    print("TRAJECTORY INFORMATION")
    print("=" * 80)
    print(f"Trajectory ID: {trajectory.get('trajectory_id', 'N/A')}")
    print(f"Total Steps: {trajectory.get('total_steps', 0)}")
    print(f"Final Reward: {trajectory.get('final_reward', 0.0):.3f}")
    print(f"Termination Reason: {trajectory.get('termination_reason', 'N/A')}")
    print(f"Created At: {trajectory.get('created_at', 'N/A')}")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="View trajectory files in chat format")
    parser.add_argument("trajectory_file", help="Path to trajectory JSON file")
    parser.add_argument("--info-only", action="store_true", 
                       help="Show only trajectory information, not the full chat format")
    parser.add_argument("--output", "-o", 
                       help="Output file path (default: stdout)")
    
    args = parser.parse_args()
    
    # Load trajectory
    trajectory = load_trajectory(args.trajectory_file)
    
    # Print trajectory information
    print_trajectory_info(trajectory)
    
    if args.info_only:
        return
    
    # Format and output
    formatted_chat = format_trajectory_chat(trajectory)
    
    if args.output:
        # Write to file
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_chat)
            print(f"Formatted trajectory saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            sys.exit(1)
    else:
        # Print to stdout
        print("CHAT FORMAT:")
        print("=" * 80)
        print(formatted_chat)


if __name__ == "__main__":
    main()
