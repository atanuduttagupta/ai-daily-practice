# PA case study — MCP (Model Context Protocol): a support-desk tool server

**Library:** the official `mcp` Python SDK, **version 2** (`MCPServer` for the server, `Client` for the client). Free and open source. No API key, no LLM needed.

## The business problem
Support agents, human or AI, keep asking the same things: "where is order X?", "can it be refunded?", "open a ticket". Instead of hard-wiring these into one chatbot, we expose them as **MCP tools**. Any MCP-capable client (Claude Desktop, an IDE, your own agent) can discover and call them with no custom integration code.

- **Input:** none to provide. The demo runs a fixed set of calls against a small in-memory order list. Edit `ORDERS` to experiment.
- **Output:** printed discovery results, tool results, an error case, a resource and a prompt.

## Run
```bash
pip install -r requirements.txt
python main.py            # demo: launches the server as a subprocess, then acts as a client
python main.py serve      # server only (what an MCP client such as Claude Desktop launches)
```
Needs Python 3.10 or newer (check `requirements.txt`; use a virtual environment).

**Version note:** many tutorials use SDK v1 (`from mcp.server.fastmcp import FastMCP`). In v2 it was renamed `MCPServer` and parts of the API changed. If you follow a v1 tutorial, install `mcp<2`.

## Theory
**MCP** is an open standard (introduced by Anthropic) for connecting AI applications to tools and data. Without it, every app needs custom code for every tool (M apps x N tools). With it, each tool is written once as an MCP **server** and each app implements one MCP **client** (M + N).

**Roles:** the **host** is the AI app (a chat app, IDE, agent); it contains an MCP **client** that connects to one or more MCP **servers**. Messages are JSON-RPC 2.0.

**Lifecycle:** connect and initialize, then *discover* (`list_tools`, `list_resources`, `list_prompts`), then *use* (`call_tool`, `read_resource`, `get_prompt`).

**The three things a server can offer:**
| Primitive | Controlled by | What it is | Example here |
|---|---|---|---|
| **Tool** | the model | An action or lookup the AI can decide to call | `check_refund_eligibility` |
| **Resource** | the application | Read-only context identified by a URI | `policy://refunds` |
| **Prompt** | the user | A reusable prompt template | `support_reply` |

**Transports:** *stdio* (the client launches the server as a local subprocess; used here) and *Streamable HTTP* (a remote server over the network, typically with authentication).

**How the pieces fit in an agent:** the LLM sees the tool names, descriptions and input schemas, decides to call one, the client executes it through MCP, and the result goes back to the LLM to continue. The SDK builds each tool's JSON schema from your Python type hints and docstring, so **clear descriptions and typed arguments matter**: the model reads them to choose tools.

**Designing tools well:**
- Keep tools small and focused, with clear names and docstrings.
- Separate read-only tools from ones that change data (`create_support_ticket`), and have a human confirm the latter.
- Return clear errors. `ToolError` is for failures you expect: the message goes back to the client and the model can react to it. Any other exception is treated as a crash (generic message to the client, traceback in the server log).

**Security basics:** tool outputs can contain text that tries to manipulate the model (**prompt injection**), so treat them as untrusted data. Give each server only the access it needs (least privilege), only run servers you trust, and authenticate remote ones.

## What the output shows
- **Discovery:** the client learned 3 tools (with their input schemas), 1 resource and 1 prompt without any hard-coded knowledge.
- **Tool calls:** A100 (delivered 15 days ago) is refund-eligible, A101 (gift card) is not (final sale), A103 (delivered 80 days ago) is past the 30-day limit. The ticket tool returned `T001`.
- **Errors:** asking for a missing order returned `is_error=True` with a helpful message instead of crashing.

The demo uses a fixed "today" (20 Nov 2026) so the day counts repeat.

## What was tested
I ran the full demo with `mcp` 2.3.0 over a real stdio subprocess. I did not test it with Claude Desktop or with an LLM calling the tools, so treat those steps as a guide. SDK details may change between versions.

## Connect it to a real client (try this)
Claude Desktop (edit its config file, use the full path to the Python that has `mcp` installed):
```json
{"mcpServers": {"support-desk": {"command": "/full/path/to/python", "args": ["/full/path/to/main.py", "serve"]}}}
```
Or explore it interactively with the MCP Inspector (needs Node.js): `npx @modelcontextprotocol/inspector python main.py serve`.

## More to try
1. Add a tool `list_orders_for_customer` and re-run the discovery step.
2. Replace the `ORDERS` dict with SQLite.
3. Serve over HTTP: `server.run("streamable-http")`.
4. Give a local Ollama model these tools and let it choose which to call (an agent loop).
5. Add a tool that rejects bad input and see how the client reports it.

## Likely interview questions
- What problem does MCP solve? How is it different from function calling?
- What are tools, resources and prompts, and who controls each?
- What are the security risks of connecting an agent to MCP servers?

## One thing learned
MCP turns "custom integration per app" into "write the tool once, any client can use it". The model only knows what your descriptions and schemas tell it, so they are part of the product.
