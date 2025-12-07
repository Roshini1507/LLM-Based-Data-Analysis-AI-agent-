# gemini_client.py
# Google AI Studio (Generative) client wrapper for text generation and embeddings.
# Assumes you store your API key in streamlit secrets: st.secrets["gcp"]["gemini_api_key"]
# and optionally st.secrets["gcp"]["gemini_default_model"] (e.g., "gemini-2.0-pro")
#
# Embedding model default: "gemini-embedding-001" (per your note)
#
#mNOTE: Google AI Studio expects the "Authorization: Bearer <API_KEY>" header.
#       Make sure your key has the Generative API enabled.

import time
import requests
import streamlit as st
import os
from typing import Optional, Any, Dict, Union, List

def _get_api_key() -> Optional[str]:
    try:
        return st.secrets["gcp"].get("gemini_api_key")
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

def _get_default_model() -> str:
    try:
        return st.secrets["gcp"].get("gemini_default_model") or "gemini-2.5-pro"
    except Exception:
        return os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-2.5-pro")

def _get_default_embedding_model() -> str:
    try:
        return st.secrets["gcp"].get("embedding_model") or "gemini-embedding-001"
    except Exception:
        return os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def _normalize_model_name_for_path(raw: str) -> str:
    # If user provided "models/gemini-2.5-pro" -> keep short name "gemini-2.5-pro"
    if raw.startswith("models/"):
        return raw.split("/", 1)[1]
    return raw

import time

def extract_text_from_gemini_response(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    Robustly extract human-readable text from Gemini JSON.
    """
    if not resp_json:
        return None

    # 1) candidates[].content.parts[].text
    try:
        if "candidates" in resp_json and isinstance(resp_json["candidates"], list) and resp_json["candidates"]:
            cand = resp_json["candidates"][0]
            content = cand.get("content") or cand.get("output") or {}
            if isinstance(content, dict) and "parts" in content and isinstance(content["parts"], list):
                pieces = [p.get("text","").strip() for p in content["parts"] if isinstance(p, dict) and p.get("text")]
                if pieces:
                    return "\n\n".join(pieces)
            if isinstance(content, list):
                pieces = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item and isinstance(item["text"], str) and item["text"].strip():
                            pieces.append(item["text"].strip())
                        elif "parts" in item:
                            for p in item["parts"]:
                                if isinstance(p, dict) and "text" in p:
                                    pieces.append(p["text"].strip())
                if pieces:
                    return "\n\n".join(pieces)
    except Exception:
        pass

    # 2) output.content list
    try:
        if "output" in resp_json and isinstance(resp_json["output"], dict):
            cont = resp_json["output"].get("content")
            if isinstance(cont, list):
                pieces = [c.get("text","").strip() for c in cont if isinstance(c, dict) and c.get("text")]
                if pieces:
                    return "\n\n".join(pieces)
    except Exception:
        pass

    # 3) fallback deep-walk for 'text' strings
    def _walk(obj):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, str) and len(v.strip()) > 3:
                    found.append(v.strip())
                else:
                    found += _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                found += _walk(it)
        return found

    pieces = _walk(resp_json)
    if pieces:
        pieces = sorted(set(pieces), key=lambda s: -len(s))
        return "\n\n".join(pieces[:3])
    return None

def generate_text(prompt: str,
                  model: Optional[str] = None,
                  temperature: float = 0.0,
                  max_output_tokens: int = 512,
                  retry_on_truncate: bool = True) -> str:
    """
    Generate text using Gemini generateContent with correct JSON shape,
    extract text robustly, and auto-retry once on MAX_TOKENS.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key not found. Set st.secrets['gcp']['gemini_api_key'] or env GEMINI_API_KEY")

    raw_model = model or _get_default_model()
    short_model = _normalize_model_name_for_path(raw_model)
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{short_model}:generateContent"

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json; charset=utf-8"}

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
            "candidateCount": 1
        }
    }

    resp = requests.post(endpoint, headers=headers, json=body, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini generateContent error {resp.status_code}: {resp.text}\nEndpoint={endpoint}")

    data = resp.json()
    # quick extract
    text = extract_text_from_gemini_response(data)

    # check finish reason
    finish = None
    try:
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            finish = data["candidates"][0].get("finishReason")
    except Exception:
        finish = None

    # retry if truncated and no text found
    if (not text) and retry_on_truncate and finish and str(finish).upper() == "MAX_TOKENS":
        # small backoff
        time.sleep(0.4)
        new_budget = min(int(max_output_tokens) * 2, 2048)
        body["generationConfig"]["maxOutputTokens"] = new_budget
        resp2 = requests.post(endpoint, headers=headers, json=body, timeout=120)
        if resp2.status_code != 200:
            raise RuntimeError(f"Gemini retry generateContent error {resp2.status_code}: {resp2.text}\nEndpoint={endpoint}")
        data2 = resp2.json()
        text = extract_text_from_gemini_response(data2)
        if text:
            return text
        # else fall through to return JSON fallback

    if text:
        return text
    # fallback: return the JSON string so you can debug in UI
    return str(data)

def embed_text(texts: Union[List[str], str], model: Optional[str] = None) -> List[List[float]]:
    """
    Request embeddings from Google AI Studio (Gemini).
    - If model is 'gemini-embedding-001' then send ONE request per input (API requires single input).
    - For other models that support batching, try to send a batch request.
    Returns: list of embedding vectors corresponding to the inputs order.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key not found. Set st.secrets['gcp']['gemini_api_key'] or env GEMINI_API_KEY")

    model_name = model or _get_default_embedding_model()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json; charset=utf-8",
    }

    # Helper to parse embedding from response JSON robustly
    def _parse_vectors_from_response(data):
        vectors = []
        if not data:
            return vectors
        # common: top-level 'embeddings' list
        if "embeddings" in data and isinstance(data["embeddings"], list):
            for e in data["embeddings"]:
                if isinstance(e, dict) and "embedding" in e:
                    vectors.append(e["embedding"])
                elif isinstance(e, list) and e and isinstance(e[0], (int, float)):
                    vectors.append(e)
            if vectors:
                return vectors
        # next: 'predictions'
        if "predictions" in data and isinstance(data["predictions"], list):
            for p in data["predictions"]:
                # find any numeric list in p
                def _find_vector(obj):
                    if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            res = _find_vector(v)
                            if res:
                                return res
                    if isinstance(obj, list):
                        for item in obj:
                            res = _find_vector(item)
                            if res:
                                return res
                    return None
                v = _find_vector(p)
                if v:
                    vectors.append(v)
            if vectors:
                return vectors
        # final fallback: deep search the whole json
        def _find_any_vector(obj):
            if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    res = _find_any_vector(v)
                    if res:
                        return res
            if isinstance(obj, list):
                for item in obj:
                    res = _find_any_vector(item)
                    if res:
                        return res
            return None
        v = _find_any_vector(data)
        if v:
            return [v]
        return vectors

    # If single string passed, normalize to list
    single_input = False
    if isinstance(texts, str):
        texts_list = [texts]
        single_input = True
    else:
        texts_list = texts

    vectors_out = []

    # If the embedding model is gemini-embedding-001 (single-input per request),
    # we must call the endpoint once per text.
    if model_name and "gemini-embedding-001" in model_name:
        for t in texts_list:
            body = {
                "model": model_name,
                "content": {"parts": [{"text": t}]}
            }
            resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini embedContent error {resp.status_code}: {resp.text}")
            data = resp.json()
            parsed = _parse_vectors_from_response(data)
            if not parsed:
                raise RuntimeError("Unable to parse embedding response: " + str(data))
            # parsed may be a list; take first vector (embedContent for single input returns single vector)
            vectors_out.append(parsed[0])
    else:
        # Try batch-friendly body using 'contents' (list of parts)
        # Each item -> {"parts":[{"text": "..."}]}
        body = {"model": model_name, "contents": [{"parts": [{"text": t}]} for t in texts_list]}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            # if the batch fails with 400 for unknown fields, attempt per-item fallback
            # so we still succeed even if endpoint doesn't like 'contents'
            if resp.status_code == 400:
                vectors_out = []
                for t in texts_list:
                    body2 = {"model": model_name, "content": {"parts": [{"text": t}]}}
                    r2 = requests.post(endpoint, headers=headers, json=body2, timeout=60)
                    if r2.status_code != 200:
                        raise RuntimeError(f"Gemini embedContent error {r2.status_code}: {r2.text}")
                    data2 = r2.json()
                    parsed2 = _parse_vectors_from_response(data2)
                    if not parsed2:
                        raise RuntimeError("Unable to parse embedding response: " + str(data2))
                    vectors_out.append(parsed2[0])
            else:
                raise RuntimeError(f"Gemini embedContent error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            parsed = _parse_vectors_from_response(data)
            if not parsed:
                # If parsed is single vector, but multiple inputs were given, try to fallback to per-item calls
                vectors_out = []
                for t in texts_list:
                    body2 = {"model": model_name, "content": {"parts": [{"text": t}]}}
                    r2 = requests.post(endpoint, headers=headers, json=body2, timeout=60)
                    if r2.status_code != 200:
                        raise RuntimeError(f"Gemini embedContent error {r2.status_code}: {r2.text}")
                    data2 = r2.json()
                    parsed2 = _parse_vectors_from_response(data2)
                    if not parsed2:
                        raise RuntimeError("Unable to parse embedding response: " + str(data2))
                    vectors_out.append(parsed2[0])
            else:
                # parsed might contain concatenated vectors; assume it matches input order
                vectors_out = parsed

    # If original input was a single string, return a list with one vector (consistent)
    if single_input:
        return vectors_out[:1]
    return vectors_out
def extract_text_from_gemini_response(resp_json):
    """
    Robustly find human-readable text in a Gemini response JSON.
    Returns a string (joined best pieces) or None if nothing found.
    """
    if not resp_json:
        return None

    # 1) candidates[].content.parts[].text (preferred)
    try:
        if "candidates" in resp_json and isinstance(resp_json["candidates"], list) and resp_json["candidates"]:
            cand = resp_json["candidates"][0]
            # candidate may have 'content' -> 'parts' -> list of dicts with 'text'
            content = cand.get("content") or cand.get("output") or {}
            if isinstance(content, dict) and "parts" in content and isinstance(content["parts"], list):
                pieces = []
                for p in content["parts"]:
                    if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                        pieces.append(p["text"].strip())
                if pieces:
                    return "\n\n".join(pieces)
            # sometimes content is list of dicts
            if isinstance(content, list):
                pieces = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            pieces.append(item["text"].strip())
                        elif "parts" in item and isinstance(item["parts"], list):
                            for p in item["parts"]:
                                if isinstance(p, dict) and "text" in p:
                                    pieces.append(p["text"].strip())
                if pieces:
                    return "\n\n".join(pieces)
    except Exception:
        pass

    # 2) output.content -> list of text blocks
    try:
        if "output" in resp_json and isinstance(resp_json["output"], dict):
            cont = resp_json["output"].get("content")
            if isinstance(cont, list):
                pieces = []
                for c in cont:
                    if isinstance(c, dict) and "text" in c and isinstance(c["text"], str):
                        pieces.append(c["text"].strip())
                if pieces:
                    return "\n\n".join(pieces)
    except Exception:
        pass

    # 3) Fallback: deep walk for any "text" fields (avoid tiny tokens)
    def _walk_find_text(obj):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, str) and len(v.strip()) > 3:
                    found.append(v.strip())
                else:
                    found += _walk_find_text(v)
        elif isinstance(obj, list):
            for item in obj:
                found += _walk_find_text(item)
        return found

    pieces = _walk_find_text(resp_json)
    if pieces:
        # return the longest pieces first joined
        pieces = sorted(set(pieces), key=lambda s: -len(s))
        return "\n\n".join(pieces[:3])

    return None
