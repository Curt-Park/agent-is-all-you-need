"""
Chapter 01. Bash Agent
==========

An autonomous agent that interacts with a bash shell using an LLM.

Mechanism (Flowchart):
----------------------

    [LLM Response]
           |
      +----+----+
      |         |
    (Tool)   (No Tool) +--------+
      |         |               |
      v         v               |
    [Bash]    [Optional HITL]   |
      |         |               |
      v         v               |
    [Update History]            |
           |                    v
         [Loop]              [Done]

How to Execute:
---------------

1. Ensure your environment is set up with mandatory environment variables
   loaded in a `.env` file (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_ID`).

2. Run the script from your terminal providing the task as an argument:

   $ python 01_build_bash_agent.py "Your instruction here"

   Optional: Specify the maximum number of bash turns:
   $ python 01_build_bash_agent.py "Your instruction here" --max-steps 10

   Optional: Enable human-in-the-loop to provide feedback:
   $ python 01_build_bash_agent.py "Your instruction here" --hitl
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# --- Constants ---

# ANSI Color Codes
COLOR_BOLD = "\033[1m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = (
    f"You are a coding agent at {os.getcwd()}. On each turn, think about what to do next and use bash to solve tasks."
)

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
    }
]

# --- Helper Functions ---


def run_bash_command(command: str) -> str:
    """Executes a shell command safely and returns captured output."""
    print(f"{COLOR_CYAN}$ {command}{COLOR_RESET}")
    try:
        # Timeout helps avoid hanging on infinite bash processes
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30)
        return result.stdout or result.stderr or "(no output generated)"
    except Exception as e:
        return f"Error executing command: {str(e)}"


def save_trajectory(messages: list[ChatCompletionMessageParam]) -> None:
    """Saves the final conversation log."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"trajectory_{timestamp}.json"
    log_file.write_text(json.dumps(messages, indent=2))
    print(f"\nTrajectory saved to: {log_file}")


# --- Main Agent Logic ---


def run_agent(task: str, max_steps: int, enable_hitl: bool) -> None:
    """Manages the agent interaction loop and logs the conversation."""
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL_ID")

    # Conversation history
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}/{max_steps} ---")

        # 1. Get agent's next action
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        messages.append(response.model_dump(exclude_none=True))

        if response.content:
            print(f"{COLOR_YELLOW}Agent says:{COLOR_RESET} {response.content}")

        # 2. Logic: Process Tool Calls
        if response.tool_calls:
            # Handle tool execution
            tool_call = response.tool_calls[0]
            if tool_call.function.name == "bash":
                command = json.loads(tool_call.function.arguments)["command"]
                output = run_bash_command(command)
                print(output)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
                continue  # Proceed to next step automatically

        # 3. Handle completion or HITL interaction
        if enable_hitl:
            print(f"\n{COLOR_MAGENTA}[HITL] Task seems complete. Provide feedback or press Enter to exit:{COLOR_RESET}")
            user_input = input()
            if user_input.strip():
                messages.append({"role": "user", "content": user_input})
                continue  # Resume loop with new user input

        print("Task marked complete by agent.")
        break

    else:
        print(f"{COLOR_RED}Reached step limit ({max_steps}).{COLOR_RESET}")

    save_trajectory(messages)


# --- Entry Point ---


def main() -> None:
    """Entry point for the bash agent script."""
    parser = argparse.ArgumentParser(description="Autonomous Bash Agent")
    parser.add_argument("task", help="The task the agent should perform.")
    parser.add_argument("--max-steps", type=int, default=30, help="Max turns in the loop.")
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop.")
    args = parser.parse_args()

    try:
        run_agent(args.task, args.max_steps, args.hitl)
    except KeyboardInterrupt:
        print("\nAgent interrupted by user.")


if __name__ == "__main__":
    main()
