from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st


DEFAULT_ENDPOINT = "http://localhost:8090/serve/sentiment"


def extract_label(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("label", "prediction", "predicted_label", "class"):
            if key in payload:
                return str(payload[key])
        if "data" in payload:
            return extract_label(payload["data"])
        if "result" in payload:
            return extract_label(payload["result"])
    if isinstance(payload, list) and payload:
        return extract_label(payload[0])
    return str(payload)


st.set_page_config(page_title="Sentiment Classifier")
st.title("Sentiment Classifier")

endpoint = st.sidebar.text_input(
    "Endpoint",
    value=os.getenv("CLEARML_SERVING_URL", DEFAULT_ENDPOINT),
)
text = st.text_area("Text", value="The service was fast and friendly.", height=140)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Enter text before prediction.")
    else:
        started = time.perf_counter()
        try:
            response = requests.post(endpoint, json={"text": text}, timeout=15)
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            st.metric("Label", extract_label(payload))
            st.metric("Latency", f"{latency_ms:.1f} ms")
            with st.expander("Raw response"):
                st.json(payload)
        except requests.exceptions.RequestException as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            st.metric("Latency", f"{latency_ms:.1f} ms")
            st.error(f"Prediction endpoint is unavailable: {exc}")
        except ValueError:
            st.error("Endpoint returned a non-JSON response.")
