import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MedRAG — Clinical Guideline Assistant", page_icon="🩺")
st.title("🩺 MedRAG: Clinical Guideline Assistant")
st.caption(
    "Answers are grounded in the indexed guideline corpus. If the retrieval "
    "confidence is too low, the system will say so instead of guessing."
)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the indexed clinical guidelines…")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

        if data["warning"]:
            st.warning(data["warning"])

        if data["abstained"]:
            st.error(data["answer"])
        else:
            st.write(data["answer"])
            st.caption(f"Top retrieval confidence: {data['top_score']:.3f}")
            with st.expander(f"Sources ({len(data['sources'])})"):
                for src in data["sources"]:
                    st.markdown(f"**{src['source_name']}** — score {src['score']}")
                    st.text(src["excerpt"])

        st.session_state.history.append({"role": "assistant", "content": data["answer"]})
