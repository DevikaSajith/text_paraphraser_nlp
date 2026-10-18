"""Streamlit demo.  Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from paraphraser import ParaphraseEngine
from paraphraser.neural import DEFAULT_MODEL

st.set_page_config(page_title="Text Paraphraser", page_icon="✍️", layout="wide")
st.title("✍️ Automatic Text Paraphraser")
st.caption("NLP course project: rule-based (WordNet + Lesk) and neural (T5) paraphrasing with automatic reranking.")


@st.cache_resource(show_spinner="Loading models…")
def get_engine(model_name: str, backend: str) -> ParaphraseEngine:
    return ParaphraseEngine(neural_model=model_name, scorer_backend=backend)


with st.sidebar:
    st.header("Settings")
    method = st.radio("Method", ["hybrid", "neural", "rule"], help="hybrid = neural + rule candidates, reranked")
    n = st.slider("Number of paraphrases", 1, 6, 3)
    strategy = st.selectbox("Neural decoding", ["diverse", "beam", "sample"])
    alpha = st.slider("Fidelity vs. novelty (α)", 0.0, 1.0, 0.6,
                      help="Higher = prefer meaning preservation; lower = prefer different wording")
    model_name = st.text_input("Model", DEFAULT_MODEL)
    backend = st.selectbox("Similarity backend", ["auto", "embedding", "bow"])

text = st.text_area("Input text", height=180,
                    value="Because the weather was terrible, we decided to cancel the picnic. "
                          "However, the children were extremely disappointed.")

if st.button("Paraphrase", type="primary") and text.strip():
    engine = get_engine(model_name, backend)
    with st.spinner("Generating…"):
        try:
            results = engine.paraphrase(text, method=method, n=n, alpha=alpha, strategy=strategy)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()
    if not results:
        st.warning("No paraphrase could be generated.")
    for i, c in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"**Option {i}** · `{c.method}`" + ("" if c.accepted else " · ⚠️ low similarity"))
            st.write(c.text)
    st.subheader("Metrics")
    st.dataframe(pd.DataFrame([
        {"method": c.method, "similarity": round(c.similarity, 3), "PINC": round(c.pinc, 3),
         "score": round(c.score, 3), "accepted": c.accepted, "text": c.text} for c in results
    ]), use_container_width=True)
    st.caption(f"Similarity backend: {engine.scorer.backend}")
