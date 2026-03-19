# Agent Is All You Need!
 
**Build, Eval, Serve, and Evolve AI Agents — from scratch to a self-personalizing production system.**
 
This project is a 15-chapter journey that teaches you to build an AI agent, rigorously evaluate each capability you add, deploy it as a production API, and continuously improve it through reinforcement learning. You start with a bare agent loop and end with a system that adapts to individual users.
 
Every line of code in this project was written using the agent built within this tutorial itself. No Claude Code, no Codex, no external coding agents — just the agent we build together, chapter by chapter.

## Philosophy
 
- **Build & Eval Together:** Every build chapter is paired with an evaluation notebook. If you can't measure it, you haven't really built it.
- **No Abstractions:** No LangChain, no LlamaIndex. All core logic is implemented from scratch so you see the *how* and *why*.
- **Production-First:** We go beyond local scripts into API servers, sandboxing, cost control, observability, and regression loops.
- **Continuous Evolution:** We collect telemetry, run A/B tests, and use RL (Bandits / DPO) to personalize agent behavior from real data.
- **Self-Hosting:** The agent we build becomes the tool we use to build the rest. Dogfooding is the ultimate eval.
 
## Roadmap
 
### Part I — Build & Eval
 
Build the agent harness step by step, evaluating every piece you add.
 
| Ch. | Build | Eval |
| :-- | :---- | :--- |
| 01 | Bash Agent | Pass / Fail |
| 02 | Multi-Tool Agent | Step-wise Efficiency |
| 03 | Planning Agent | Plan Quality (LLM-as-Judge) |
| 04 | Subagent | Multi-Agent Attribution |
| 05 | Skills Agent | A/B Test (Skill Impact) |
| 06 | Context Compact | Compression Quality |
| 07 | Tasks (Persistence) | Dependency Correctness |
| 08 | Background Tasks | Throughput & Concurrency |
 
### Part II — Serve & Verify
 
Wrap the agent in a production API and build the feedback loops.
 
| Ch. | Focus |
| :-- | :---- |
| 09 | API Server + SSE Streaming + Docker Sandboxing |
| 10 | Cost Control + Model Routing + Semantic Caching |
| 11 | Observability + Replay + Regression Loop |
 
### Part III — Learn & Personalize
 
Turn production traces into training signal. Make the agent evolve.
 
| Ch. | Focus |
| :-- | :---- |
| 12 | Trajectory Collection + Heuristic Personalization |
| 13 | Contextual Bandit for Tool Selection |
| 14 | DPO for Agent Behavior Alignment |
| 15 | Closed-Loop Personalization Pipeline |
 
## Key Concepts
 
**Part I (Ch.01–08)** builds the agent harness — execution loop, tool dispatch, planning, subagents, skills, context compression, persistent tasks, and background execution — while teaching you to *evaluate every piece you add*.
 
**Part II (Ch.09–11)** wraps that agent in a FastAPI server with streaming, sandboxing, cost optimization, and a feedback loop that turns production failures into regression tests.
 
**Part III (Ch.12–15)** closes the loop: traces become trajectories, trajectories become training signal, and RL (Bandits → DPO) turns a generic agent into one that adapts to each user.
 
## Getting Started
 
```bash
# 1. Clone
git clone https://github.com/Curt-Park/agent-is-all-you-need.git
cd agent-is-all-you-need
 
# 2. Install mise (https://mise.jdx.dev)
curl https://mise.run | sh
 
# 3. Set up Python env
mise trust && mise install && uv sync
 
# 4. Configure API key
cp .env.example .env   # fill in your key — works with any OpenAI-compatible API
 
# 5. Start
python ch_01_build_bash_agent.py

# 6. Run tests
pytest tests/test_bash_agent.py
```
 
## Credits
 
Inspired by [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) (shareAI-lab) — an excellent tutorial for agent harness construction. This project extends that foundation into rigorous evaluation, production serving, and RL-based personalization.
 
## License
 
MIT