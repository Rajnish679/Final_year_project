
import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Agent Chatbot")
st.write("Create and interact with the AI Agents!")

# Input Fields
system_prompt = st.text_area("System Prompt:", height=80, placeholder="You are a helpful AI tutor...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow web search")

user_query = st.text_area("Enter your query:", height=150, placeholder="Ask Anything")

API_URL = "http://127.0.0.1:8503/chat"

if st.button("Ask Agent!"):
    if not user_query.strip():
        st.error("Please type a question.")
    else:
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }
        with st.spinner("Waiting for response…"):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.markdown("**AI:** " + data["response"])
            except Exception as e:
                st.error(f"Request failed: {e}")










































































