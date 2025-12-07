# indexer.py
import os
import time
import concurrent.futures

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from gemini_client import embed_text
from typing import List, Dict, Any

# Conservative batch sizes to avoid Pinecone 2MB request limit
BATCH_SIZE = 32
CHUNK_UPSERT = 16
RETRY_ATTEMPTS = 2

class Indexer:
    def __init__(self):
        # Read Pinecone config from streamlit secrets or environment
        try:
            pine_key = st.secrets["pinecone"]["api_key"]
            index_name = st.secrets["pinecone"]["index_name"]
        except Exception:
            pine_key = os.environ.get("PINECONE_API_KEY")
            index_name = os.environ.get("PINECONE_INDEX", "data-analysis-rag-agent")

        if not pine_key:
            raise RuntimeError("Pinecone API key missing. Add to .streamlit/secrets.toml or environment variable PINECONE_API_KEY.")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=pine_key)
        self.index_name = index_name

        # Detect embedding dimension using embed_text on a small sample
        try:
            sample_vecs = embed_text("dim detection sample")
            if not sample_vecs or not isinstance(sample_vecs, list):
                raise RuntimeError("Embedding provider returned no vectors for dimension detection.")
            detected_dim = len(sample_vecs[0])
        except Exception as e:
            raise RuntimeError(f"Failed to detect embedding dimension: {e}")

        # Check existing indexes and ensure correct dimension
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if index_name in existing_indexes:
            # Try to inspect index metadata/dimension if available
            existing_dim = None
            try:
                meta = self.pc.describe_index(index_name)
                if isinstance(meta, dict):
                    existing_dim = meta.get("dimension")
            except Exception:
                existing_dim = None

            # If dimension mismatch, delete and recreate index
            if existing_dim and existing_dim != detected_dim:
                try:
                    self.pc.delete_index(index_name)
                    existing_indexes.remove(index_name)
                except Exception as e:
                    # If delete fails, surface informative error
                    raise RuntimeError(f"Pinecone index dimension mismatch (existing: {existing_dim}, expected: {detected_dim}) and failed to delete: {e}")

        # Create index if not exists
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=detected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # small wait for index to be provisioned
            time.sleep(1.0)

        # Connect to index
        self.index = self.pc.Index(index_name)

    def index_dataframe(self, name: str, df, batch_size: int = BATCH_SIZE):
        """
        Index a pandas DataFrame by converting each row to a short text doc and upserting vectors.
        Metadata keys use 'source' and 'row_id' so rag.py can read them.
        """
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for i, (_, row) in enumerate(df.iterrows()):
            row_dict = row.to_dict()
            # Build compact text representation for the row
            parts = [f"{k}:{v}" for k, v in row_dict.items()]
            text = f"{name} | row {i} | " + " | ".join(parts)
            if len(text) > 2000:
                text = text[:2000]

            texts.append(text)
            metas.append({"source": name, "row_id": i})
            ids.append(f"{name.replace('.', '_')}_row_{i}")

            if len(texts) >= batch_size:
                self._upsert_batch(ids, texts, metas)
                texts = []; metas = []; ids = []

        # final flush
        if texts:
            self._upsert_batch(ids, texts, metas)

    def index_text_chunks(self, name: str, chunks: List[str], batch_size: int = BATCH_SIZE):
        """
        Index textual chunks (e.g., from a PDF). Metadata keys use 'source' and 'chunk_index'.
        """
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        texts: List[str] = []

        for i, c in enumerate(chunks):
            txt = c if len(c) <= 2000 else c[:2000]
            ids.append(f"{name.replace('.', '_')}_chunk_{i}")
            metas.append({"source": name, "chunk_index": i})
            texts.append(txt)

            if len(texts) >= batch_size:
                self._upsert_batch(ids, texts, metas)
                ids = []; metas = []; texts = []

        if texts:
            self._upsert_batch(ids, texts, metas)


    # you can tune these
    EMBED_WORKERS = 4   # concurrency for embedding calls (reduce for free tier)
    CHUNK_UPSERT = 16   # keep this small for Pinecone 2MB safety; you can increase to 32 if payloads are small

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Efficiently get embeddings for a list of texts.
        - If the embedding model supports batching (handled by your embed_text), call it once.
        - If gemini-embedding-001 (single-input), call embed_text(t) concurrently per text.
        Returns list of vectors in the same order as texts.
        """
        # Attempt a single batched call first (embed_text will fallback when model requires single-input)
        try:
            vecs = embed_text(texts)
            if isinstance(vecs, list) and len(vecs) == len(texts):
                return vecs
        except Exception:
            # fallback to per-item concurrent calls
            pass

        # Per-item concurrent embedding (slower per-call but parallel)
        vecs_out = [None] * len(texts)
        def _embed_one(idx, t):
            # embed_text accepts single string and returns a list or single vector depending on implementation
            res = embed_text(t)
            # res may be [[v]] or [v], so normalize:
            if isinstance(res, list) and len(res)>0 and isinstance(res[0], list):
                return res[0]
            if isinstance(res, list) and isinstance(res[0], (float, int)):
                return res
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
            futures = {ex.submit(_embed_one, i, t): i for i, t in enumerate(texts)}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    vecs_out[i] = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Embedding failed for index {i}: {e}")

        # sanity check
        for v in vecs_out:
            if not v or not isinstance(v, list):
                raise RuntimeError("Embedding returned invalid result in _embed_texts.")
        return vecs_out


    def _upsert_batch(self, ids: List[str], texts: List[str], metas: List[Dict[str, Any]], retry: int = RETRY_ATTEMPTS, show_progress: bool = True):
        """
        Embed texts and upsert to Pinecone in safe-sized chunks.
        - Uses concurrent embedding when needed.
        - Shows Streamlit progress while working.
        - Skips items already present in index if possible (optional).
        """
        if not ids or not texts or not metas:
            return

        # Optionally skip IDs that already exist to save time (useful on re-index)
        try:
            # fetch only the first 1000 ids present check; adjust if small dataset
            # Note: .fetch may not accept a very large list; use it cautiously or skip if not desired.
            existing_map = {}
            try:
                fetched = self.index.fetch(ids=ids)
                # fetched may contain 'vectors' mapping
                if isinstance(fetched, dict) and "vectors" in fetched:
                    existing_map = {k: v for k, v in fetched["vectors"].items()}
            except Exception:
                existing_map = {}
        except Exception:
            existing_map = {}

        # Which items actually need embedding/upsert
        need_indices = [i for i, _id in enumerate(ids) if ids[i] not in existing_map]

        if not need_indices:
            # nothing to upsert
            return

        # Prepare to process only needed items
        ids_to_do = [ids[i] for i in need_indices]
        texts_to_do = [texts[i] for i in need_indices]
        metas_to_do = [metas[i] for i in need_indices]

        # Progress bar
        progress = None
        if show_progress:
            try:
                progress = st.progress(0)
                total_steps = len(ids_to_do)
                st.write(f"Indexing {total_steps} items (this may take a while)...")
            except Exception:
                progress = None

        # 1) Get embeddings (concurrent if necessary)
        vecs = self._embed_texts(texts_to_do)

        # 2) Build upsert items with truncated text in metadata
        upserts = []
        for _id, vec, meta, txt in zip(ids_to_do, vecs, metas_to_do, texts_to_do):
            truncated = txt if len(txt) <= 500 else txt[:500]
            new_meta = dict(meta)
            new_meta["text"] = truncated
            upserts.append({"id": _id, "values": vec, "metadata": new_meta})

        # 3) Upsert in manageable chunks
        done = 0
        for i in range(0, len(upserts), CHUNK_UPSERT):
            part = upserts[i:i+CHUNK_UPSERT]
            attempt = 0
            while attempt <= retry:
                try:
                    self.index.upsert(vectors=part)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > retry:
                        raise RuntimeError(f"Pinecone upsert failed after {retry} retries: {e}")
                    time.sleep(0.5 * attempt)
            done += len(part)
            if progress:
                try:
                    progress.progress(min(100, int(done / len(upserts) * 100)))
                except Exception:
                    pass
            # small sleep to avoid bursts
            time.sleep(0.05)

        # finalize progress
        if progress:
            try:
                progress.progress(100)
            except Exception:
                pass

    def query(self, query_text: str, top_k: int = 5):
        """
        Embed the query_text and query Pinecone. Returns list of matches in the form:
        [{"id": ..., "score": ..., "meta": {...}}, ...]
        """
        try:
            vecs = embed_text(query_text)
            if not vecs or not isinstance(vecs, list):
                raise RuntimeError("Embedding returned no vectors.")
            query_vec = vecs[0]

            # Query Pinecone - include metadata for source/row mapping
            resp = self.index.query(vector=query_vec, top_k=top_k, include_metadata=True, include_values=False)

            matches = []
            # resp may be a dict with 'matches' key
            if isinstance(resp, dict) and "matches" in resp:
                for m in resp["matches"]:
                    matches.append({
                        "id": m.get("id"),
                        "score": m.get("score"),
                        "meta": m.get("metadata", {})
                    })
                return matches

            # resp may be object-like with .matches attribute
            if hasattr(resp, "matches"):
                for m in resp.matches:
                    if isinstance(m, dict):
                        meta = m.get("metadata", {}) or m.get("meta", {}) or {}
                        matches.append({"id": m.get("id"), "score": m.get("score"), "meta": meta})
                    else:
                        matches.append({"id": getattr(m, "id", None), "score": getattr(m, "score", None), "meta": getattr(m, "metadata", {})})
                return matches

            # fallback: try to iterate resp
            try:
                for m in resp:
                    matches.append({
                        "id": m.get("id"),
                        "score": m.get("score"),
                        "meta": m.get("metadata", {})
                    })
                return matches
            except Exception:
                return []
        except Exception as e:
            raise RuntimeError(f"Indexer.query failed: {e}")
# indexer.py
import os
import time
import streamlit as st
import concurrent.futures
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from gemini_client import embed_text

# -----------------------------
# CONFIG (ALL VARIABLES DEFINED)
# -----------------------------
BATCH_SIZE = 32                 # how many rows to accumulate before upsert
CHUNK_UPSERT = 16               # upsert chunk size (must be small due to Pinecone free-tier)
RETRY_ATTEMPTS = 2              # retry on upsert failure
EMBED_WORKERS = 4               # number of threads for embedding parallelism
MAX_TEXT_LEN = 500              # store first N chars of row text in metadata

class Indexer:
    def __init__(self):
        # get Pinecone keys
        try:
            pine_key = st.secrets["pinecone"]["api_key"]
            index_name = st.secrets["pinecone"]["index_name"]
        except Exception:
            pine_key = os.environ.get("PINECONE_API_KEY")
            index_name = os.environ.get("PINECONE_INDEX", "data-analysis-rag-agent")

        if not pine_key:
            raise RuntimeError("Pinecone API key missing.")

        self.pc = Pinecone(api_key=pine_key)
        self.index_name = index_name

        # ---- auto-detect embedding dimension ----
        try:
            vec = embed_text("dimension test")[0]
            detected_dim = len(vec)
        except Exception as e:
            raise RuntimeError(f"Embedding dim detection failed: {e}")

        # create or recreate index
        existing = [i["name"] for i in self.pc.list_indexes()]
        if index_name in existing:
            # try to fetch existing dimension
            try:
                desc = self.pc.describe_index(index_name)
                existing_dim = desc.get("dimension")
            except Exception:
                existing_dim = None

            # mismatch? recreate
            if existing_dim and existing_dim != detected_dim:
                self.pc.delete_index(index_name)
                existing.remove(index_name)

        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=detected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(1)

        self.index = self.pc.Index(index_name)

    # ---------------------------------------------------
    # PARALLEL EMBEDDING WRAPPER
    # ---------------------------------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed all texts. If the embedding model supports batching via embed_text,
        use it. If not, fallback to parallel single calls.
        """
        try:
            # try batch
            batched = embed_text(texts)
            if isinstance(batched, list) and len(batched) == len(texts):
                return batched
        except Exception:
            pass

        # fallback — parallel single-input embedding
        results = [None] * len(texts)

        def _do(idx, t):
            out = embed_text(t)
            if isinstance(out, list) and isinstance(out[0], list):
                return out[0]
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
            fut_map = {ex.submit(_do, i, t): i for i, t in enumerate(texts)}
            for fut in concurrent.futures.as_completed(fut_map):
                idx = fut_map[fut]
                results[idx] = fut.result()

        return results

    # ---------------------------------------------------
    # MAIN UPSERT FUNCTION
    # ---------------------------------------------------
    def _upsert_batch(self, ids: List[str], texts: List[str], metas: List[Dict[str, Any]]):
        if not ids:
            return

        # ---- PROGRESS DISPLAY ----
        try:
            progress = st.progress(0)
        except Exception:
            progress = None

        # ---- EMBEDDINGS ----
        vecs = self._embed_texts(texts)

        upserts = []
        for _id, vec, meta, txt in zip(ids, vecs, metas, texts):
            snippet = txt if len(txt) <= MAX_TEXT_LEN else txt[:MAX_TEXT_LEN]
            meta2 = dict(meta)
            meta2["text"] = snippet
            upserts.append({"id": _id, "values": vec, "metadata": meta2})

        # ---- UPSERT IN CHUNKS ----
        total = len(upserts)
        done = 0

        for i in range(0, total, CHUNK_UPSERT):
            part = upserts[i:i + CHUNK_UPSERT]

            attempt = 0
            while attempt <= RETRY_ATTEMPTS:
                try:
                    self.index.upsert(vectors=part)
                    break
                except Exception:
                    attempt += 1
                    if attempt > RETRY_ATTEMPTS:
                        raise
                    time.sleep(0.5 * attempt)

            done += len(part)
            if progress:
                try:
                    progress.progress(int(done / total * 100))
                except:
                    pass

            time.sleep(0.05)

        if progress:
            try:
                progress.progress(100)
            except:
                pass

    # ---------------------------------------------------
    # INDEXING DATAFRAME
    # ---------------------------------------------------
    def index_dataframe(self, name: str, df):
        texts, metas, ids = [], [], []

        for i, (_, row) in enumerate(df.iterrows()):
            # convert row compactly
            row_text = f"{name} | row {i} | " + " | ".join([f"{k}:{v}" for k, v in row.to_dict().items()])
            row_text = row_text[:2000]

            texts.append(row_text)
            metas.append({"source": name, "row_id": i})
            ids.append(f"{name}_row_{i}".replace(".", "_"))

            if len(texts) >= BATCH_SIZE:
                self._upsert_batch(ids, texts, metas)
                texts, metas, ids = [], [], []

        if texts:
            self._upsert_batch(ids, texts, metas)

    # ---------------------------------------------------
    # INDEXING TEXT CHUNKS (PDF, DOC, etc)
    # ---------------------------------------------------
    def index_text_chunks(self, name: str, chunks: List[str]):
        texts, metas, ids = [], [], []

        for i, c in enumerate(chunks):
            txt = c[:2000]
            texts.append(txt)
            metas.append({"source": name, "chunk_index": i})
            ids.append(f"{name}_chunk_{i}".replace(".", "_"))

            if len(texts) >= BATCH_SIZE:
                self._upsert_batch(ids, texts, metas)
                texts, metas, ids = [], [], []

        if texts:
            self._upsert_batch(ids, texts, metas)

    # ---------------------------------------------------
    # QUERY
    # ---------------------------------------------------
    def query(self, query_text: str, top_k: int = 5):
        try:
            vec = embed_text(query_text)[0]
        except Exception as e:
            raise RuntimeError(f"Query embedding failed: {e}")

        resp = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        matches = []
        if isinstance(resp, dict) and "matches" in resp:
            for m in resp["matches"]:
                matches.append({
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "meta": m.get("metadata", {})
                })
        elif hasattr(resp, "matches"):
            for m in resp.matches:
                matches.append({
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "meta": getattr(m, "metadata", {})
                })

        return matches
