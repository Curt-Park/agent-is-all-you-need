"""
Chapter 02. Multi-Tool Agent
============================

Extends the bash agent with a DuckDuckGo search tool, allowing the agent to
gather external information in addition to file system manipulation.

Changes from Chapter 01 (Bash Agent):
------------------------------------
    1. Added `ddgs` library dependency for web searching.
    2. Updated `SYSTEM_PROMPT` to acknowledge the new search capability.
    3. Expanded `TOOLS` list to include `websearch` functionality.
    4. Implemented `perform_websearch` helper function.
    5. Updated the agent's main tool execution loop to dispatch between
       `bash` and `websearch` based on the LLM's requested function call.

Usage:
------
    $ python ch_02_multi_tool_agent.py "What time is it in Seoul now?"
    $ python ch_02_multi_tool_agent.py "Search the web for Python 3.13 new features and summarize them" --max-steps 10
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS

# Initialize client
load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))

# Define model, prompts, and tools
MODEL = os.getenv("LLM_MODEL_ID")
SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use the provided tools (bash, websearch) to solve tasks."

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
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Perform a web search using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
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

def perform_websearch(query: str) -> str:
    """Performs a web search using DuckDuckGo."""
    print(f"\033[92m[Searching web] Query: {query}\033[0m")
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error performing search: {e}"

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

        # Execute if tools were requested
        if response.tool_calls:
            for tool_call in response.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                output = ""
                if func_name == "bash":
                    output = run_bash_command(func_args["command"])
                elif func_name == "websearch":
                    output = perform_websearch(func_args["query"])
                
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
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop")
    args = parser.parse_args()

    run_agent(args.task, args.max_steps, args.hitl)

if __name__ == "__main__":
    main()
