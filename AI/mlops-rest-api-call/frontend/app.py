
import streamlit as st
import httpx

# Cloudflare-deployed FastAPI backend
API_BASE_URL = "https://fastapi-prompt-store.atanu-duttagupta.workers.dev"

@st.cache_data(ttl=60)
def check_backend_health():
    try:
        response = httpx.get(
            f"{API_BASE_URL}/health",
            timeout=10
        )
        return response.status_code == 200
    except httpx.RequestError:
        return False

    
st.set_page_config(
    page_title="Prompt Store",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Prompt Store")
st.caption("Manage and explore your AI prompts")

# Check backend health
if check_backend_health():
    st.success("Backend connected")
else:
    st.error("Backend is not responding")


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Create Prompt",
    "View Prompts",
    "Get Prompt",
    "Token Estimator"
])


# 1. Create Prompt
with tab1:
    st.subheader("Create a new prompt")

    prompt_text = st.text_area("Prompt text", max_chars=2000)
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
                    f"{API_BASE_URL}/prompts",
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
                    st.error(f"API Error: {response.text}")

            except httpx.RequestError as e:
                st.error(f"Connection error: {e}")


# 2. View Prompts
with tab2:
    st.subheader("All saved prompts")

    limit = st.number_input(
        "Maximum prompts to display",
        min_value=1,
        max_value=100,
        value=10
    )

    if st.button("Load Prompts"):
        try:
            response = httpx.get(
                f"{API_BASE_URL}/prompts",
                params={"limit": limit},
                timeout=10
            )

            if response.status_code == 200:
                prompts = response.json()

                if prompts:
                    st.dataframe(prompts, use_container_width=True)
                else:
                    st.info("No prompts found.")

            else:
                st.error(f"API Error: {response.text}")

        except httpx.RequestError as e:
            st.error(f"Connection error: {e}")


# 3. Get Prompt by ID
with tab3:
    st.subheader("Retrieve a prompt")

    prompt_id = st.number_input(
        "Enter Prompt ID",
        min_value=1,
        step=1
    )

    if st.button("Get Prompt"):
        try:
            response = httpx.get(
                f"{API_BASE_URL}/prompts/{prompt_id}",
                timeout=10
            )

            if response.status_code == 200:
                st.json(response.json())
            elif response.status_code == 404:
                st.warning("Prompt not found.")
            else:
                st.error(f"API Error: {response.text}")

        except httpx.RequestError as e:
            st.error(f"Connection error: {e}")


# 4. Token Estimator
with tab4:
    st.subheader("Estimate prompt tokens")

    text_to_estimate = st.text_area("Enter text to estimate")

    if st.button("Estimate Tokens"):
        if not text_to_estimate.strip():
            st.warning("Please enter some text.")
        else:
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/estimate-tokens",
                    json={"text": text_to_estimate},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.metric(
                        "Estimated Tokens",
                        result["estimated_tokens"]
                    )
                    st.write("Characters:", result["characters"])
                else:
                    st.error(f"API Error: {response.text}")

            except httpx.RequestError as e:
                st.error(f"Connection error: {e}")
