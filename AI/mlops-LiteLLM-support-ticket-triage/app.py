
import streamlit as st

from main import (
    GEMINI_MODEL,
    GROQ_MODEL,
    SYSTEM,
    TICKETS,
    build_cloud_router,
    parse_category,
)


st.set_page_config(
    page_title="Support Ticket Triage",
    page_icon="🎫",
    layout="centered",
)


@st.cache_resource
def get_router(primary_model=None):
    return build_cloud_router(primary_model=primary_model)


st.title("🎫 Support Ticket Triage")
st.caption("Powered by LiteLLM, Groq and Gemini with automatic model fallback")

st.divider()

# Cloud routing configuration
st.subheader("Model Configuration")

test_fallback = st.toggle(
    "Simulate Groq primary failure",
    value=False,
    help="Uses an invalid Groq model to test automatic fallback to Gemini.",
)

if test_fallback:
    st.warning("Fallback testing mode: Groq primary is deliberately unavailable.")
    st.write("**Primary model:** Invalid Groq model (test)")
else:
    st.success("Normal cloud mode: Groq primary with automatic Gemini fallback.")
    st.write(f"**Primary model:** {GROQ_MODEL}")

st.write(f"**Backup model:** {GEMINI_MODEL}")

st.divider()


# Ticket input
st.subheader("Enter Support Ticket")


def update_ticket_input():
    st.session_state.ticket_input = st.session_state.sample_ticket


# Initialize text area state
if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

sample_tickets = [""] + [ticket for ticket, category in TICKETS]

selected_ticket = st.selectbox(
    "Choose an example ticket (optional)",
    sample_tickets,
    format_func=lambda x: x if x else "Select a sample ticket...",
    key="sample_ticket",
    on_change=update_ticket_input,
)

ticket = st.text_area(
    "Or enter your own ticket",
    height=120,
    placeholder="Enter support ticket description...",
    key="ticket_input",
)

submitted = st.button(
    "Classify Ticket",
    type="primary",
    use_container_width=True,
)

if submitted:
    if not ticket or not ticket.strip():
        st.warning("Please enter or select a support ticket.")
        st.stop()

    with st.spinner("Classifying ticket..."):
        try:
            if test_fallback:
                router = get_router(
                    primary_model="groq/invalid-model-for-fallback-test"
                )
            else:
                router = get_router()

            response = router.completion(
                model="triage-primary",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ticket},
                ],
                temperature=0,
                max_tokens=100,
            )

            reply = response.choices[0].message.content or ""
            category = parse_category(reply)

            served_model = response.model or ""

            if "gemini" in served_model.lower():
                served_by = f"Backup ({GEMINI_MODEL})"
            else:
                served_by = f"Primary ({GROQ_MODEL})"

            st.divider()
            st.subheader("Classification Result")

            st.success(f"Category: {category.upper()}")

            st.write(f"**Served by:** {served_by}")

            if "gemini" in served_model.lower():
                st.info("Groq primary failed. Automatic Gemini fallback succeeded.")

        except Exception as error:
            st.error(f"Classification failed: {error}")

st.divider()

st.caption("AI Daily Practice Project | LiteLLM MLOps Case Study")
