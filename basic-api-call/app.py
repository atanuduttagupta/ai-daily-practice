import streamlit as st
from main import (
    ask_llm,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMQuotaError,
    LLMRequestError,
    LLMUnavailableError,
)

st.title("Basic LLM API Call")

st.write("Ask a question and get a response from LLM.")


question = st.text_area(
    "Your Question",
    placeholder="Enter Your Question Here..."
)

token_length = st.slider(
    "Maximum output tokens",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

if st.button("Ask LLM"):
    if not question.strip():
        st.warning("Please enter a question here...")
    else:

        try:

            with st.spinner("LLM is thinking..."):
                answer = ask_llm(question, token_length)

            st.subheader("LLM's Answer")
            st.write(answer)

        except LLMQuotaError as error:
            st.warning(str(error))

        except LLMUnavailableError as error:
            st.warning(str(error))

        except LLMAuthenticationError as error:
            st.error(str(error))

        except LLMConfigurationError as error:
            st.error(str(error))

        except LLMRequestError as error:
            st.warning(str(error))