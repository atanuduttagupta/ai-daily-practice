import streamlit as st
import httpx

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Prompt Store",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Prompt Store")
st.caption("A simple Streamlit frontend powered by FastAPI")

# Check backend connection
try:
    response = httpx.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    st.success("FastAPI backend connected")
except httpx.RequestError:
    st.error("Cannot connect to FastAPI. Please start the backend first.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Create Prompt",
    "List Prompts",
    "Get Prompt",
    "Estimate Tokens"
])

# 1. Create Prompt
with tab1:
    st.subheader("Create a new prompt")

    prompt_text = st.text_area("Prompt text", height=150)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    if st.button("Save Prompt"):
        if not prompt_text.strip():
            st.warning("Please enter prompt text.")
        else:
            try:
                response = httpx.post(
                    f"{API_URL}/prompts",
                    json={
                        "text": prompt_text,
                        "temperature": temperature
                    },
                    timeout=10
                )

                if response.status_code == 201:
                    st.success("Prompt created successfully!")
                    st.json(response.json())
                else:
                    st.error(f"API error: {response.text}")

            except httpx.RequestError as error:
                st.error(f"Connection error: {error}")

# 2. List Prompts
with tab2:
    st.subheader("All saved prompts")

    limit = st.number_input(
        "Maximum prompts",
        min_value=1,
        max_value=100,
        value=10
    )

    if st.button("Load Prompts"):
        try:
            response = httpx.get(
                f"{API_URL}/prompts",
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()

            prompts = response.json()

            if prompts:
                st.dataframe(prompts, use_container_width=True)
            else:
                st.info("No prompts found.")

        except httpx.HTTPStatusError as error:
            st.error(f"API error: {error.response.text}")
        except httpx.RequestError as error:
            st.error(f"Connection error: {error}")

# 3. Get Prompt
with tab3:
    st.subheader("Retrieve a prompt")

    prompt_id = st.number_input(
        "Prompt ID",
        min_value=1,
        step=1
    )

    if st.button("Get Prompt"):
        try:
            response = httpx.get(
                f"{API_URL}/prompts/{prompt_id}",
                timeout=10
            )

            if response.status_code == 200:
                st.json(response.json())
            elif response.status_code == 404:
                st.warning("Prompt not found.")
            else:
                st.error(f"API error: {response.text}")

        except httpx.RequestError as error:
            st.error(f"Connection error: {error}")

# 4. Estimate Tokens
with tab4:
    st.subheader("Estimate token count")

    text_to_estimate = st.text_area(
        "Enter text",
        height=150,
        key="token_text"
    )

    if st.button("Estimate Tokens"):
        if not text_to_estimate.strip():
            st.warning("Please enter some text.")
        else:
            try:
                response = httpx.post(
                    f"{API_URL}/estimate-tokens",
                    json={"text": text_to_estimate},
                    timeout=10
                )

                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(f"API error: {response.text}")

            except httpx.RequestError as error:
                st.error(f"Connection error: {error}")