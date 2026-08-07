import asyncio
import os
import time
import random
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import requests # Dùng requests cho ổn định

st.set_page_config(page_title="AHI-Orchestrator", layout="wide", page_icon="🧬")

# --- KẾT NỐI DB NEON ---
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
    except: pass

init_neon_tables()

# --- LẤY KEYS ---
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

MODELS = {
    "AHI-Gemini": {"p": "gemini", "m": "gemini-1.5-flash"},
    "AHI-Grok": {"p": "groq", "m": "llama-3.3-70b-versatile"},
    "AHI-ChatGPT": {"p": "openrouter", "m": "openai/gpt-4o-mini"},
    "AHI-Claude": {"p": "openrouter", "m": "anthropic/claude-3.5-sonnet"},
    "AHI-DeepSeek": {"p": "openrouter", "m": "deepseek/deepseek-r1"},
    "AHI-Llama": {"p": "openrouter", "m": "meta-llama/llama-3.3-70b-instruct"},
    "AHI-Qwen": {"p": "openrouter", "m": "qwen/qwen-2.5-72b-instruct"},
    "AHI-MistralLarge": {"p": "openrouter", "m": "mistralai/mistral-large-2411"}
}

def fetch_status(ahi_p):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT total_answers, expert_code FROM expert_state WHERE ahi_p = %s;", (ahi_p,))
    res = cur.fetchone()
    if not res:
        cur.execute("INSERT INTO expert_state (ahi_p, expert_code, total_answers) VALUES (%s, %s, %s) RETURNING *;", (ahi_p, "AHI-P-EXPERT-01", 0))
        conn.commit()
        res = {"total_answers": 0, "expert_code": "AHI-P-EXPERT-01"}
    return dict(res)

def save_data(ahi_p, name, prompt, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE expert_state SET total_answers = total_answers + 1 WHERE ahi_p = %s;", (ahi_p,))
    vec = [random.uniform(-1, 1) for _ in range(1536)]
    cur.execute("INSERT INTO ahi_evolution_vectors (ahi_p, ahi_name, prompt, content, embedding) VALUES (%s, %s, %s, %s, %s);", (ahi_p, name, prompt, content, vec))
    conn.commit()

def call_ai(name, cfg, prompt):
    try:
        if cfg["p"] == "gemini":
            url = f"https://googleapis.com{cfg['m']}:generateContent?key={GEMINI_KEY}"
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        
        base = "https://groq.com" if cfg["p"] == "groq" else "https://openrouter.ai"
        key = GROQ_KEY if cfg["p"] == "groq" else OPENROUTER_KEY
        r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": cfg["m"], "messages": [{"role": "user", "content": prompt}]})
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Lỗi gọi AI: {str(e)}"

st.title("🧬 AHI-Orchestrator Workspace")

user_id = st.sidebar.text_input("Mã AHI-P:", value="AHI-P-EXPERT-01")
if user_id:
    status = fetch_status(user_id)
    st.sidebar.write(f"**Tích lũy: {status['total_answers']}**")

prompt = st.text_input("Nhập câu lệnh (Nhấn ENTER):")

if prompt:
    st.write("---")
    for name, cfg in MODELS.items():
        res = call_ai(name, cfg, prompt)
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            with st.expander(f"📌 {name}", expanded=True): st.write(res)
        with col2:
            if st.button("Lưu", key=name):
                save_data(user_id, name, prompt, res)
                st.success("Đã lưu!")
                time.sleep(0.5)
                st.rerun()
