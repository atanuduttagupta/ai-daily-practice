"""
PA case study - MCP (Model Context Protocol): a support-desk tool server for AI assistants
Library: official `mcp` Python SDK, v2 (MCPServer for the server, Client for the client).

Business problem: a support agent (human or AI) keeps asking the same questions: "where is
order X?", "can it be refunded?", "open a ticket". Instead of hard-wiring these into one chatbot,
we expose them as MCP tools. ANY MCP-capable client (Claude Desktop, an IDE, your own agent)
can then discover and call them with no custom integration.

This one file is both halves:
    python main.py            -> demo: starts the server as a subprocess, then acts as a client
    python main.py serve      -> just the server (stdio), for MCP clients to launch
No API key, no LLM, no cost: the demo calls the tools directly so you can see the protocol.

Setup:  pip install -r requirements.txt
"""

import asyncio
import sys
from datetime import date

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

TODAY = date(2026, 11, 20)   # fixed "today" so the demo gives the same answers every time

# A tiny in-memory "database" (a real server would query a database or an API).
ORDERS = {
    "A100": {"item": "Wireless headphones", "status": "delivered", "delivered": date(2026, 11, 5), "final_sale": False},
    "A101": {"item": "Gift card", "status": "delivered", "delivered": date(2026, 11, 10), "final_sale": True},
    "A102": {"item": "Laptop stand", "status": "shipped", "delivered": None, "final_sale": False},
    "A103": {"item": "USB-C cable", "status": "delivered", "delivered": date(2026, 9, 1), "final_sale": False},
}
TICKETS: list[dict] = []
REFUND_POLICY = ("Refunds: items can be returned within 30 days of delivery. "
                 "Final-sale items (such as gift cards) cannot be refunded. "
                 "Orders that have not been delivered cannot be refunded yet.")

server = MCPServer("support-desk", instructions="Tools for looking up orders and handling support requests.")


def get_order(order_id: str) -> dict:
    order = ORDERS.get(order_id.upper())
    if order is None:
        raise ToolError(f"Order '{order_id}' not found. Known orders look like A100.")   # expected failure: the message reaches the client
    return order


# ------------------------------------------------------------------ server side
@server.tool()
def get_order_status(order_id: str) -> str:
    """Look up the item and delivery status of an order."""
    o = get_order(order_id)
    when = f" on {o['delivered']}" if o["delivered"] else ""
    return f"Order {order_id.upper()}: {o['item']} is {o['status']}{when}."


@server.tool()
def check_refund_eligibility(order_id: str) -> str:
    """Decide whether an order can be refunded under the refund policy."""
    o = get_order(order_id)
    if o["final_sale"]:
        return "Not eligible: final-sale item."
    if o["delivered"] is None:
        return "Not eligible yet: the order has not been delivered."
    days = (TODAY - o["delivered"]).days
    return (f"Eligible: delivered {days} days ago (limit 30)." if days <= 30
            else f"Not eligible: delivered {days} days ago, past the 30-day limit.")


@server.tool()
def create_support_ticket(order_id: str, issue: str) -> str:
    """Open a support ticket for an order. Changes data, so clients should confirm with a human."""
    get_order(order_id)
    ticket = {"id": f"T{len(TICKETS) + 1:03d}", "order_id": order_id.upper(), "issue": issue}
    TICKETS.append(ticket)
    return f"Created ticket {ticket['id']} for order {ticket['order_id']}."


@server.resource("policy://refunds")
def refund_policy() -> str:
    """The refund policy text. Resources are read-only context an AI can load."""
    return REFUND_POLICY


@server.prompt()
def support_reply(order_id: str) -> str:
    """A reusable prompt template: draft a polite reply about an order."""
    return (f"Look up order {order_id}, check refund eligibility, and write a short, polite reply to "
            f"the customer. Use only facts returned by the tools.")


# ------------------------------------------------------------------ client side
async def demo() -> None:
    params = StdioServerParameters(command=sys.executable, args=[__file__, "serve"])
    async with Client(params) as client:
        print(f"Connected to server: {client.server_info.name}\n")

        print("=== 1) Discovery: what can this server do? ===")
        for tool in (await client.list_tools()).tools:
            print(f"tool    {tool.name}: {tool.description}")
            print(f"        input schema properties: {list(tool.input_schema['properties'])}")
        for res in (await client.list_resources()).resources:
            print(f"resource {res.uri}")
        for prompt in (await client.list_prompts()).prompts:
            print(f"prompt  {prompt.name}")

        print("\n=== 2) Calling tools ===")
        for name, args in [("get_order_status", {"order_id": "A102"}),
                           ("check_refund_eligibility", {"order_id": "A100"}),
                           ("check_refund_eligibility", {"order_id": "A101"}),
                           ("check_refund_eligibility", {"order_id": "A103"}),
                           ("create_support_ticket", {"order_id": "A100", "issue": "Left ear cup is crackling"})]:
            result = await client.call_tool(name, args)
            print(f"{name}({args}) -> {result.content[0].text}")

        print("\n=== 3) Errors are returned, not crashes ===")
        bad = await client.call_tool("get_order_status", {"order_id": "Z999"})
        print(f"is_error={bad.is_error}: {bad.content[0].text}")

        print("\n=== 4) Reading a resource and a prompt ===")
        print("policy://refunds ->", (await client.read_resource("policy://refunds")).contents[0].text)
        prompt = await client.get_prompt("support_reply", {"order_id": "A100"})
        print("support_reply prompt ->", prompt.messages[0].content.text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        server.run()          # stdio transport: waits for an MCP client on stdin/stdout
    else:
        asyncio.run(demo())
