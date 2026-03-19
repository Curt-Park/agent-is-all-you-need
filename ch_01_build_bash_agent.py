"""
Chapter 01. Bash Agent
======================

A simple autonomous coding agent that interacts with a bash shell.

Mechanism:
----------
    1. Agent receives a task.
    2. Agent decides: Need a tool?
       - Yes -> Execute bash, return output to agent, loop.
       - No  -> Task complete or seek human feedback.
    3. Loop until task done or max steps reached.

Usage:
------
    $ python ch_01_build_bash_agent.py "List current files"
    $ python ch_01_build_bash_agent.py "Create a README.md" --max-steps 5 --hitl
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Initialize client with env vars
load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
MODEL = os.getenv("LLM_MODEL_ID")

# Define prompts and tools for the LLM
SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def run_bash_command(command: str) -> str:
    """Executes a shell command and returns the output."""
    print(f"\033[96m$ {command}\033[0m")
    try:
        res = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30)
        return res.stdout or res.stderr or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def run_agent(task: str, max_steps: int = 10, enable_hitl: bool = False) -> None:
    """Core agent loop: orchestrates the LLM turns and tool execution."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        # Get decision from LLM
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS).choices[0].message
        messages.append(response.model_dump(exclude_none=True))

        if response.content:
            print(f"\033[93mAgent:\033[0m {response.content}")

        # Execute bash if requested
        if response.tool_calls:
            for tool_call in response.tool_calls:
                command = json.loads(tool_call.function.arguments)["command"]
                output = run_bash_command(command)
                print(output)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            continue

        # Check for task completion / human feedback
        if enable_hitl:
            feedback = input("\033[95m[HITL] Provide feedback (or press Enter to finish): \033[0m")
            if feedback.strip():
                messages.append({"role": "user", "content": feedback})
                continue

        print("Task marked complete.")
        break

    # Save final conversation
    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / f"trajectory_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(messages, indent=2))
    print(f"\nTrajectory saved: {log_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task to perform")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop")
    args = parser.parse_args()

    run_agent(args.task, args.max_steps, args.hitl)


if __name__ == "__main__":
    main()
