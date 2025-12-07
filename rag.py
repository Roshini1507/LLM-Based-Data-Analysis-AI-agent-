from gemini_client import generate_text
import streamlit as st
from indexer import Indexer
import json

# RAG Agent that uses Indexer to retrieve and Gemini to generate answers
class RAGAgent:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model_name = model_name
        try:
            self.indexer = st.session_state["indexer"]
            if self.indexer is None:
                self.indexer = Indexer()
        except Exception:
            self.indexer = Indexer()

    def answer_question(self, question: str, top_k: int = 2):
        """
        Answer question by:
         1) Building a short dataset summary (if available)
         2) Asking LLM directly with that summary (LLM-first, fewer retrievals)
         3) If needed, include top_k retrieved rows (their 'text' metadata) and ask LLM again (grounded)
        Returns: (answer_text, sources_list, plot_data)
        """
        import pandas as pd
        # 0) Try to build a short dataset summary from session (if available)
        dataset_summaries = []
        try:
            uploaded = st.session_state.get("uploaded_data", {}) or {}
            for src_name, df in uploaded.items():
                try:
                    nrows, ncols = df.shape
                    numeric = df.select_dtypes(include=["number"]).columns.tolist()
                    sample_stats = ""
                    if numeric:
                        # add mean of first numeric column as a hint
                        col = numeric[0]
                        sample_stats = f"{col} mean {df[col].mean():.2f}, median {df[col].median():.2f}."
                    dataset_summaries.append(f"{src_name}: {nrows} rows × {ncols} cols. {sample_stats}")
                except Exception:
                    continue
        except Exception:
            dataset_summaries = []

        dataset_summary_text = ""
        if dataset_summaries:
            dataset_summary_text = "Dataset summaries: " + " || ".join(dataset_summaries)

        # 1) LLM-first attempt (use summary only to reduce RAG reliance)
        prompt_llm_first = "You are a data assistant. Answer the user's question using the dataset summary provided. If you must assume any value, state the assumption.\n\n"
        if dataset_summary_text:
            prompt_llm_first += dataset_summary_text + "\n\n"
        prompt_llm_first += f"Question: {question}\nAnswer concisely. If data not sufficient, say 'I don't know based on the provided data.'"

        try:
            llm_answer = generate_text(prompt_llm_first, model=self.model_name, temperature=0.0, max_output_tokens=400)
        except Exception as e:
            llm_answer = f"Generation error: {e}"

        # If LLM answer appears informative (not the 'I don't know' fallback), return it
        if isinstance(llm_answer, str) and "I don't know based on the provided data" not in llm_answer:
            return llm_answer, [], None

        # 2) Otherwise, do retrieval to ground LLM (include the retrieved 'text' metadata)
        matches = self.indexer.query(question, top_k=top_k)
        context_texts = []
        sources = []
        for m in matches:
            meta = m.get("meta", {})
            src = meta.get("source", "unknown")
            idx = meta.get("row_id") if meta.get("row_id") is not None else meta.get("chunk_index")
            sources.append(f"{src}:{idx}")
            # prefer the 'text' field we saved in metadata; fall back to ID marker if absent
            row_text = meta.get("text")
            if row_text:
                # include small header so LLM knows the source
                context_texts.append(f"[{src}:{idx}] {row_text}")
            else:
                context_texts.append(f"[{src}:{idx}] (row text not available)")

        # Build grounded prompt
        prompt = "You are a data assistant. Use ONLY the provided context to answer. Cite sources like [source:row].\n\n"
        if dataset_summary_text:
            prompt += dataset_summary_text + "\n\n"
        if context_texts:
            prompt += "Context rows:\n" + "\n---\n".join(context_texts) + "\n\n"
        prompt += f"Question: {question}\nAnswer concisely and cite sources."

        try:
            ans = generate_text(prompt, model=self.model_name, temperature=0.0, max_output_tokens=1024)
        except Exception as e:
            ans = f"Generation error: {e}"

        # Try to detect a plot request (same as before)
        plot_data = None
        if "plot" in question.lower() or ("show" in question.lower() and "over" in question.lower()):
            try:
                if st.session_state.get("uploaded_data"):
                    first = list(st.session_state["uploaded_data"].values())[0]
                    plot_data = {"df": first, "kind": "line", "x": first.columns[0], "y": first.columns[1] if len(first.columns)>1 else first.columns[0]}
            except Exception:
                plot_data = None

        return ans, sources, plot_data
