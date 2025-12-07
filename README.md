# Data Analysis RAG Agent (Streamlit + Gemini + Pinecone)

This project is a Streamlit-based Retrieval-Augmented-Generation (RAG) Data Analysis Agent that:
- Accepts CSV / Excel / JSON / PDF uploads
- Generates proactive insights on upload (automatic)
- Indexes document chunks / table rows into Pinecone using embeddings (Gemini embeddings)
- Answers user questions via RAG using Gemini models (user-selectable)
- Provides visualizations, follow-up question suggestions, and downloadable summary

> **Important**: You must supply your own Gemini API key and Pinecone API key. Place them into `.streamlit/secrets.toml` (see template file included).

## Files
- `app.py` - Streamlit app entrypoint
- `parsers.py` - File parsing utilities (csv/xlsx/json/pdf)
- `insights.py` - Proactive insights generator (stats, outliers, trends)
- `indexer.py` - Chunking + embedding + Pinecone upsert code
- `rag.py` - Retrieval + RAG QA logic (LangChain-style wrappers)
- `gemini_client.py` - Lightweight Gemini HTTP wrappers (generate + embeddings) — **edit endpoints if needed**
- `viz.py` - plotting helpers
- `config/secrets_template.toml` - template for Streamlit secrets file
- `.streamlit/secrets.toml` - **DO NOT COMMIT** — placeholder file included here for local testing (edit with your keys)
- `requirements.txt` - pip dependencies

## How to run locally
1. Install requirements:
```bash
python -m pip install -r requirements.txt
```
2. Edit `.streamlit/secrets.toml` and put your keys (see template).
3. Run the app:
```bash
streamlit run app.py
```

## Notes about Gemini usage
This project uses a simple HTTP wrapper (`gemini_client.py`) targeted at Google's Generative Language API endpoints as the default Gemini interface. Depending on the exact Gemini API you have access to, you may need to adjust the endpoints and the request shape inside `gemini_client.py` and `embeddings` functions. The code includes comments showing where to change model names and endpoints.

## Security
- **Do not commit actual keys to a public repo.** Use the `.streamlit/secrets.toml` or environment variables in production.
- For Streamlit Cloud, add your keys in the App's Secrets settings rather than committing them.
