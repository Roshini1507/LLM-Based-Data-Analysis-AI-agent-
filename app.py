import streamlit as st
from parsers import parse_file_to_df_or_docs
from insights import generate_proactive_insights
from indexer import Indexer
from rag import RAGAgent
from viz import plot_df_sample
import pandas as pd
import os

st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")

st.title("Data Analysis AI Agent")

# Sidebar: model selection and secrets check
st.sidebar.header("Configuration")
MODEL_DISPLAY = ["Gemini 2.5 Pro (balanced)", "Gemini 2.5 Flash (fast)"]
MODEL_CHOICES =     ["gemini-2.5-pro", "gemini-2.5-flash"]
model_choice = st.sidebar.selectbox("Select Gemini model", MODEL_CHOICES)
selected_index = st.selectbox("Choose model", range(len(MODEL_DISPLAY)), format_func=lambda i: MODEL_DISPLAY[i])
selected_model_id = MODEL_CHOICES[selected_index]
st.session_state["selected_model"] = selected_model_id

st.sidebar.markdown("**Pinecone** and **Gemini** keys must be set in `.streamlit/secrets.toml`.")
if st.sidebar.button("Show config (for debug)"):
    try:
        st.sidebar.write(st.secrets.to_dict())
    except Exception as e:
        st.sidebar.error("No secrets found. Add `.streamlit/secrets.toml`.")

# File uploader
uploaded_files = st.file_uploader("Upload CSV / Excel / JSON / PDF", accept_multiple_files=True, type=["csv","xlsx","xls","json","pdf"])

if "indexer" not in st.session_state:
    st.session_state["indexer"] = None
if "rag" not in st.session_state:
    st.session_state["rag"] = None

if uploaded_files:
    st.info("Parsing uploaded files...")
    docs = []
    dfs = {}
    for up in uploaded_files:
        res = parse_file_to_df_or_docs(up)
        if isinstance(res, pd.DataFrame):
            dfs[up.name] = res
        else:
            docs.append({"name": up.name, "chunks": res})

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Data Preview / Documents")
        for name, df in dfs.items():
            st.markdown(f"**{name}** — {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(10))
        for d in docs:
            st.markdown(f"**{d['name']}** — {len(d['chunks'])} text chunks (from PDF)")
            for c in d["chunks"][:3]:
                st.write(c[:500] + ("..." if len(c)>500 else ""))

    with col2:
        st.subheader("Proactive Insights")
        insights_summary = {}
        for name, df in dfs.items():
            ins = generate_proactive_insights(df, max_findings=6)
            st.markdown(f"**Insights — {name}**")
            for b in ins["bullets"]:
                st.write("- " + b)
            if ins.get("followups"):
                st.write("**Suggested follow-ups**:")
                for f in ins["followups"]:
                    if st.button(f"{name} ▸ {f}"):
                        st.session_state.setdefault("last_query", f)
            insights_summary[name] = ins

    # Initialize indexer and RAG agent (idempotent)
    if st.session_state["indexer"] is None:
        st.session_state["indexer"] = Indexer()
    indexer: Indexer = st.session_state["indexer"]

    # Index tab
    st.markdown("---")
    if st.button("Index uploaded data to Pinecone (create/update)"):
        with st.spinner("Indexing..."):
            # Index dataframes as row-documents
            for name, df in dfs.items():
                indexer.index_dataframe(name, df)
            for d in docs:
                indexer.index_text_chunks(d["name"], d["chunks"])
        st.success("Indexing finished.")

# Q&A section
st.markdown("---")
st.header("Ask a question about your uploaded data (RAG)")
query = st.text_input("Enter your question here", value=st.session_state.get("last_query",""))

colq1, colq2 = st.columns([3,1])
with colq1:
    if st.button("Ask"):
        if st.session_state.get("indexer") is None:
            st.error("No indexed data found. Upload files and index them first.")
        else:
            if st.session_state.get("rag") is None:
                st.session_state["rag"] = RAGAgent(model_name=model_choice)
            rag: RAGAgent = st.session_state["rag"]
            with st.spinner("Retrieving and generating answer..."):
                answer, sources, plot_data = rag.answer_question(query, top_k=4)
            st.markdown("### Answer")
            st.write(answer)
            if sources:
                st.markdown("**Sources**")
                for s in sources:
                    st.write(f"- {s}")
            if plot_data is not None:
                st.plotly_chart(plot_df_sample(plot_data["df"], plot_data["kind"], plot_data["x"], plot_data["y"]), use_container_width=True)

with colq2:
    st.markdown("Quick prompts")
    if st.button("Show suggested follow-ups"):
        last_insights = None
        # show followups from last indexed dataframe if present
        if insights_summary:
            last_insights = list(insights_summary.values())[-1]
        if last_insights and last_insights.get("followups"):
            for f in last_insights["followups"]:
                if st.button(f"Run: {f}"):
                    st.session_state["last_query"] = f
                    st.experimental_rerun()

st.markdown("---")
st.caption("Notes: This is a demo agent. Make sure to add your Gemini & Pinecone keys in .streamlit/secrets.toml. Gemini endpoints in gemini_client.py may need adjusting depending on your access.")
