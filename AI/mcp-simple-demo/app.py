
import asyncio
import sys
from pathlib import Path

import streamlit as st
from mcp import Client, StdioServerParameters


# ---------------------------------------------------------
# MCP client connection
# ---------------------------------------------------------

SERVER_FILE = Path(__file__).with_name("main.py")


async def run_mcp_request(action: str, arguments: dict | None = None):
    """Launch the MCP server, connect, perform one request, and disconnect."""

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_FILE), "serve"],
    )

    async with Client(params) as client:

        if action == "discover":
            tools = (await client.list_tools()).tools
            resources = (await client.list_resources()).resources
            prompts = (await client.list_prompts()).prompts

            return {
                "server": client.server_info.name,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputs": tool.input_schema.get("properties", {}),
                    }
                    for tool in tools
                ],
                "resources": [str(resource.uri) for resource in resources],
                "prompts": [prompt.name for prompt in prompts],
            }

        if action == "tool":
            result = await client.call_tool(
                arguments["name"],
                arguments["args"],
            )
            messages = [
                item.text
                for item in result.content
                if hasattr(item, "text")
            ]
            return {
                "is_error": result.is_error,
                "text": "\n".join(messages) or "No text returned.",
            }

        if action == "resource":
            result = await client.read_resource("policy://refunds")
            return result.contents[0].text

        if action == "prompt":
            result = await client.get_prompt(
                "support_reply",
                {"order_id": arguments["order_id"]},
            )
            return result.messages[0].content.text


def mcp_request(action: str, arguments: dict | None = None):
    """Synchronous wrapper for Streamlit."""
    return asyncio.run(run_mcp_request(action, arguments))


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title="MCP Support Desk",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ MCP Support Desk")
st.caption(
    "An interactive demonstration of Model Context Protocol "
    "using a real MCP client and server."
)

st.info(
    "This application communicates with the support-desk MCP server "
    "through the stdio transport. It does not call the Python functions "
    "directly."
)

# ---------------------------------------------------------
# Server discovery
# ---------------------------------------------------------

with st.expander("🔍 MCP Server Discovery", expanded=True):

    if st.button("Discover MCP capabilities"):
        try:
            st.session_state["discovery"] = mcp_request("discover")
        except Exception as exc:
            st.error(f"Could not connect to MCP server: {exc}")

    discovery = st.session_state.get("discovery")

    if discovery:
        st.success(f"Connected to: {discovery['server']}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Tools", len(discovery["tools"]))
        col2.metric("Resources", len(discovery["resources"]))
        col3.metric("Prompts", len(discovery["prompts"]))

        st.subheader("Available tools")

        for tool in discovery["tools"]:
            with st.expander(tool["name"]):
                st.write(tool["description"])
                st.json(tool["inputs"])

        st.write("**Resources:**", discovery["resources"])
        st.write("**Prompts:**", discovery["prompts"])


# ---------------------------------------------------------
# Support desk operations
# ---------------------------------------------------------

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📦 Order Status",
        "💰 Refund Eligibility",
        "🎫 Create Ticket",
        "📖 Refund Policy",
    ]
)


def show_tool_result(tool_name: str, arguments: dict):
    """Call a discovered MCP tool and display its response."""
    try:
        result = mcp_request(
            "tool",
            {
                "name": tool_name,
                "args": arguments,
            },
        )

        if result["is_error"]:
            st.error(result["text"])
        else:
            st.success(result["text"])

    except Exception as exc:
        st.error(f"MCP request failed: {exc}")


# ---------------------------------------------------------
# 1. Order status
# ---------------------------------------------------------

with tab1:

    st.subheader("Check Order Status")

    order_id = st.text_input(
        "Enter Order ID",
        value="A102",
        key="status_order",
    )

    if st.button("Check Status", key="status_button"):
        if order_id.strip():
            show_tool_result(
                "get_order_status",
                {"order_id": order_id.strip()},
            )
        else:
            st.warning("Please enter an order ID.")


# ---------------------------------------------------------
# 2. Refund eligibility
# ---------------------------------------------------------

with tab2:

    st.subheader("Check Refund Eligibility")

    refund_order = st.selectbox(
        "Select an order",
        ["A100", "A101", "A102", "A103", "Z999"],
    )

    if st.button("Check Eligibility"):
        show_tool_result(
            "check_refund_eligibility",
            {"order_id": refund_order},
        )


# ---------------------------------------------------------
# 3. Create support ticket
# ---------------------------------------------------------

with tab3:

    st.subheader("Create Support Ticket")

    ticket_order = st.selectbox(
        "Order ID",
        ["A100", "A101", "A102", "A103"],
        key="ticket_order",
    )

    issue = st.text_area(
        "Describe the issue",
        placeholder="Example: Left ear cup is crackling",
    )

    confirm_ticket = st.checkbox(
        "I confirm that I want to create this support ticket."
    )

    if st.button("Create Ticket"):

        if not issue.strip():
            st.warning("Please describe the issue.")

        elif not confirm_ticket:
            st.warning("Please confirm before creating the ticket.")

        else:
            show_tool_result(
                "create_support_ticket",
                {
                    "order_id": ticket_order,
                    "issue": issue.strip(),
                },
            )


# ---------------------------------------------------------
# 4. Refund policy resource
# ---------------------------------------------------------

with tab4:

    st.subheader("Refund Policy Resource")

    st.write(
        "This information is retrieved from the MCP resource "
        "`policy://refunds`, not from an MCP tool."
    )

    if st.button("Read Refund Policy"):
        try:
            policy = mcp_request("resource")
            st.info(policy)
        except Exception as exc:
            st.error(f"Could not read resource: {exc}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Learning project | Python MCP SDK v2 | "
    "MCP Tools · Resources · Prompts · stdio Transport"
)
