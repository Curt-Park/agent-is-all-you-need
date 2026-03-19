"""
Build a Bash Agent
==================

This module implements a simple autonomous agent that uses an LLM to interact
with a bash shell to accomplish tasks.

Mechanism (Flowchart):
----------------------

    [Start]
       |
       v
    [Initialize Chat] (System Prompt + Initial Task)
       |
       +-----> [LLM Generates Response/Tool Call]
       |             |
       |             v
       |       [Is a Tool Call Requested?] --(No)--> [Finish]
       |             |
       |            (Yes)
       |             |
       |             v
       |       [Execute Bash Command]
       |             |
       |             v
       |       [Append Output to History]
       |             |
       +-------------+
       |
 [Repeat until Task done or Max Steps reached]

How to Execute:
---------------

1. Ensure your environment is set up with mandatory environment variables
   loaded in a `.env` file (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_ID`).

2. Run the script from your terminal providing the task as an argument:

   $ python 01_build_bash_agent.py "Your instruction here"

   Optional: Specify the maximum number of bash turns:
   $ python 01_build_bash_agent.py "Your instruction here" --max-steps 10
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Load environment variables (API keys, model configuration)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# The system prompt dictates the agent's persona, its current working directory (cwd), and its core capabilities.
SYSTEM = f"You are a coding agent at {os.getcwd()}. On each turn, think about what to do next and use bash to solve tasks."


def run_bash(command: str) -> str:
    """Executes a shell command safely and returns the output."""
    print(f"\033[96m$ {command}\033[0m")
    try:
        # Run the command with a timeout to prevent infinite loops
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30)
        return result.stdout or result.stderr or "(no output)"
    except Exception as e:
        return str(e)


# Define the tools available to the LLM. Currently, only 'bash' is supported.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def run_agent(task: str, max_steps: int) -> None:
    """Initializes the chat, manages the agent loop, and tracks the conversation trajectory."""
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL_ID")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]

    # Main agent loop
    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}/{max_steps} ---")
        # Request completion from the agent (which includes potential tool calls)
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        messages.append(response.model_dump(exclude_none=True))

        if response.content:
            print(f"\033[93mResponse:\033[0m {response.content}")

        # If there are no tool calls, the agent has finished its task
        if not response.tool_calls:
            break

        # Execute the requested tool (bash command)
        tool_call = response.tool_calls[0]
        func = tool_call.function
        if func.name == "bash":
            args = json.loads(func.arguments)
            output = run_bash(args["command"])
            print(output)
            # Send the tool output back into the chat conversation
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

    # Save the full interaction history for debugging/analysis
    log = Path("logs") / f"trajectory_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(messages, indent=2))
    print(f"\nTrajectory saved to {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="The programming task for the agent to complete.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum number of turns to take.")
    args = parser.parse_args()
    run_agent(args.task, args.max_steps)
