import asyncio
import os
import time
import random
from typing import Dict, Any
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import httpx

st.set_page_config(page_title="AHI-Orchestrator Workspace", layout="wide", page_icon="🧬")

def get_connection():
    return psycopg2.connect(st.secrets["NEON_DB_URL"])

def init_neon_tables():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE TABLE IF NOT EXISTS expert_state (ahi_p TEXT PRIMARY KEY, expert_code TEXT NOT NULL, total_answers INTEGER DEFAULT 0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        cur.execute("CREATE TABLE IF NOT EXISTS ahi_evolution_vectors (id SERIAL PRIMARY KEY, ahi_p TEXT REFERENCES expert_state(ahi_p), ahi_name TEXT NOT NULL, prompt TEXT NOT NULL, content TEXT NOT NULL, embedding vector(1536), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Lỗi DB: {e}")

init_neon_tables()

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

AHI_OLD_MODELS = {
    "AHI-Gemini": {"provider": "gemini", "model": "gemini-1.5-flash"},
    "AHI-Grok": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "AHI-ChatGPT": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
    "AHI-Claude": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"},
    "AHI-DeepSeek": {"provider": "openrouter", "model": "deepseek/deepseek-r1"},
    "AHI-Llama": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
    "AHI-Qwen": {"provider": "openrouter", "model": "qwen/qwen-2.5-72b-instruct"},
    "AHI-MistralLarge": {"provider": "openrouter", "model": "mistralai/mistral-large-2411"}
}

def fetch_expert_status(ahi_p: str) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT total_answers, expert_code FROM expert_state WHERE ahi_p = %s;", (ahi_p,))
    res = cur.fetchone()
    if not res:
        cur.execute("INSERT INTO expert_state (ahi_p, expert_code, total_answers) VALUES (%s, %s, %s) RETURNING *;", (ahi_p, "AHI-P-EXPERT-01", 0))
        conn.commit()
        res = {"total_answers": 0, "expert_code": "AHI-P-EXPERT-01"}
    cur.close()
    conn.close()
    return dict(res)

def save_single(ahi_p: str, ahi_name: str, prompt: str, content: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE expert_state SET total_answers = total_answers + 1, updated_at = NOW() WHERE ahi_p = %s;", (ahi_p,))
    mock_vec = [randomuniform(-) for _ in range()]
    cur.execute("INSERT INTO ahi_evolution_vectors (ahi_p, ahi_name, prompt, content, embedding) VALUES (%s, %s, %s, %s, %s);", (ahi_p, ahi_name, prompt, content, mock_vec))
    conn.commit()
    cur.close()
    conn.close()

async def fetch_ai(name: str, cfg: Dict[str, str], prompt: str) -> str:
    prov = cfg["provider"]
    mod = cfg["model"]
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if prov == "gemini":
                # Ép gọi API trực tiếp với header khóa AQ.
                url = f"https://googleapis.com{mod}:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    return res.json()["candidates"]["content"]["parts"]["text"]
                return f"Lỗi Gemini ({res.status_code}): {res.text[:]}"
            
            base = "https://groq.com" if prov == "groq" else "https://openrouter.ai"
            key = GROQ_KEY if prov == "groq" else OPENROUTER_KEY
            res = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": mod, "messages": [{"role": ""content": prompt}]})
            if res.status_code == 200:
                return res.json()["choices"]["message"]["content"]
            return f"Lỗi {prov} ({res.status_code}): {res.text[:]}"
        except Exception as e:
            return f"Lỗi hệ thống: {str(e)}"

async def run_all(prompt: str):
    tasks = [fetch_ai(n, c, prompt) for n, c in AHI_OLD_MODELSitems()]
    return await asyncio.gather(*tasks)

st.title("🧬 AHI-Orchestrator Workspace")
st.caption("Đồng bộ Đám mây Neon PostgreSQL (DBRS + DBV)")

if "results" not in st.session_state: st.session_state.results = None
if "last_p" not in st.session_state: st.session_state.last_p = ""

user_id = st.sidebar.text_input("Mã AHI-P:", value="AHI-P-EXPERT-01")
if user_id:
    status = fetch_expert_status(user_id)
    st.sidebar.write(f"Kết nối: `{status['expert_code']}`")
    st.sidebar.write(f"**Tích lũy: {status['total_answers']}**")

prompt = st.text_input("Nhập câu lệnh (Nhấn ENTER):", key="input_box")

if prompt and prompt != st.session_state.last_p:
    st.session_state.last_p = prompt
    st.session_state.results = asyncio.run(run_all(prompt))

if st.session_state.results:
    st.write("---")
    model_names = list(AHI_OLD_MODELS.keys())
    for i, res_text in enumerate(st.session_state.results):
        m_name = model_names[i]
        c1, c2 = st.columns()
        with c1:
            with st.expander(f"📌 {m_name}", expanded=True): st.write(res_text)
        with c2:
            if st.button("Lưu", key=f"btn_{i}"):
                save_single(user_id, m_name, st.session_state.last_p, res_text)
                st.success("Đã lưu!")
                time.sleep(0.5)
                st.rerun()
