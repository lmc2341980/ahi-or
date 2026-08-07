import asyncio
import json
import os
import time
import math
import random
from typing import Dict, Any, List, Tuple
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import httpx

st.set_page_config(page_title="AHI-Orchestrator Engine", layout="wide", page_icon="🧬")

# --- KẾT NỐI DATABASE NEON ---
def get_connection():
    db_url = st.secrets["NEON_DB_URL"]
    return psycopg2.connect(db_url)

def init_neon_tables():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS expert_state (
                ahi_p TEXT PRIMARY KEY,
                expert_code TEXT NOT NULL,
                total_answers INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ahi_evolution_vectors (
                id SERIAL PRIMARY KEY,
                ahi_p TEXT REFERENCES expert_state(ahi_p),
                ahi_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Lỗi khởi tạo Database: {e}")

init_neon_tables()

# Đọc cấu hình bảo mật API Keys
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

AHI_OLD_MODELS = {
    "AHI-Gemini": {"provider": "gemini", "model": "gemini-2.5-flash"},
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
    try:
        cur.execute("SELECT total_answers, expert_code FROM expert_state WHERE ahi_p = %s;", (ahi_p,))
        result = cur.fetchone()
        if not result:
            cur.execute("INSERT INTO expert_state (ahi_p, expert_code, total_answers) VALUES (%s, %s, %s) RETURNING *;", (ahi_p, "AHI-P-EXPERT-01", 0))
            conn.commit()
            return {"total_answers": 0, "expert_code": "AHI-P-EXPERT-01"}
        return dict(result)
    finally:
        cur.close()
        conn.close()

def mock_generate_embedding() -> List[float]:
    return [random.uniform(-1, 1) for _ in range(1536)]

def save_single_passed_evolution(ahi_p: str, ahi_name: str, prompt: str, content: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE expert_state SET total_answers = total_answers + 1, updated_at = NOW() WHERE ahi_p = %s;", (ahi_p,))
        vector_data = mock_generate_embedding()
        cur.execute("INSERT INTO ahi_evolution_vectors (ahi_p, ahi_name, prompt, content, embedding) VALUES (%s, %s, %s, %s, %s);", (ahi_p, ahi_name, prompt, content, vector_data))
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi lưu trữ đám mây Neon: {e}")
    finally:
        cur.close()
        conn.close()

async def fetch_single_ahi_old(ahi_name: str, config: Dict[str, str], prompt: str) -> str:
    provider = config["provider"]
    model = config["model"]
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            # 1. XỬ LÝ LUỒNG GEMINI KHÓA AQ.
            if provider == "gemini":
                if not GEMINI_KEY or GEMINI_KEY == "dummy_key":
                    return "❌ Chưa điền GEMINI_API_KEY trong mục Secrets của Streamlit."
                # VÁ LỖI CÚ PHÁP: Thêm dấu gạch chéo xuôi '/' trước dấu hai chấm để định vị đúng tài nguyên
                url = f"https://googleapis.com{model}/:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                res = await client.post(url, headers=headers, json=data)
                
                if res.status_code == 200:
                    return res.json()["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return f"❌ Lỗi Gemini (Mã {res.status_code}): Khóa API Key hết hạn mức gọi miễn phí hoặc chưa kích hoạt."

            # 2. XỬ LÝ LUỒNG GROQ
            elif provider == "groq":
                if not GROQ_KEY or GROQ_KEY == "dummy_key":
                    return "❌ Chưa điền GROQ_API_KEY trong mục Secrets."
                url = "https://groq.com"
                headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
                data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                res = await client.post(url, headers=headers, json=data)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
                else:
                    return f"❌ Lỗi Groq (Mã {res.status_code}): Tài khoản hết tiền hoặc khóa API bị lỗi."

            # 3. XỬ LÝ LUỒNG OPENROUTER (6 MÔ HÌNH CÒN LẠI)
            elif provider == "openrouter":
                if not OPENROUTER_KEY or OPENROUTER_KEY == "dummy_key":
                    return "❌ Chưa điền OPENROUTER_API_KEY trong mục Secrets."
                url = "https://openrouter.ai"
                headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
                data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                res = await client.post(url, headers=headers, json=data)
                
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
                else:
                    return f"❌ Lỗi OpenRouter (Mã {res.status_code}): Số dư tài khoản bằng 0, không thể gọi được {ahi_name}."
            
        except Exception as e:
            return f"⚠️ Lỗi kết nối luồng xử lý: {str(e)}"

    return "❌ Trục trặc hệ thống phân cấp."

async def generate_all_responses(prompt: str) -> Dict[str, str]:
    tasks = {name: fetch_single_ahi_old(name, cfg, prompt) for name, cfg in AHI_OLD_MODELS.items()}
    names = list(tasks.keys())
    res_list = await asyncio.gather(*tasks.values())
    return {names[i]: res_list[i] for i in range(len(names))}

# ==============================================================================
# GIAO DIỆN HIỂN THỊ STREAMLIT WORKSPACE
# ==============================================================================
st.title("🧬 AHI-Orchestrator Workspace & Multi-AI Dispatcher")
st.caption("Cấu trúc lưu trữ đồng bộ Đám mây Neon PostgreSQL (DBRS + DBV)")

if "current_results" not in st.session_state:
    st.session_state["current_results"] = None
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = ""
if "db_answers_count" not in st.session_state:
    st.session_state["db_answers_count"] = 0

st.sidebar.header("👤 Định Danh Người Dùng (AHI-P)")
user_id = st.sidebar.text_input("Mã định danh AHI-P:", value="AHI-P-EXPERT-01")

if user_id:
    try:
        user_status = fetch_expert_status(user_id)
        st.session_state["db_answers_count"] = user_status['total_answers']
        st.sidebar.markdown(f"""
        - Đã kết nối AHI-P: `{user_status['expert_code']}`
        - **Tổng câu trả lời tích lũy thực tế: {st.session_state["db_answers_count"]}**
        """)
    except Exception as db_err:
        st.sidebar.error(f"Lỗi DB: {db_err}")

# Ô nhập liệu nhấn ENTER kích hoạt siêu tốc
main_prompt = st.text_input("Nhập câu lệnh điều phối kiến thức (Gõ xong nhấn ENTER chạy luôn):", value=st.session_state["last_prompt"])

if main_prompt and main_prompt != st.session_state["last_prompt"]:
    st.session_state["last_prompt"] = main_prompt
    with st.spinner("Hệ thống đang truy vấn đa luồng dữ liệu siêu tốc..."):
        st.session_state["current_results"] = asyncio.run(generate_all_responses(main_prompt))

if st.button("🚀 Kích hoạt Truy Vấn Đa Mô Hình Thủ Công"):
    if main_prompt.strip():
        st.session_state["last_prompt"] = main_prompt
        with st.spinner("Hệ thống đang truy vấn đa luồng dữ liệu..."):
            st.session_state["current_results"] = asyncio.run(generate_all_responses(main_prompt))

# HIỂN THỊ KẾT QUẢ VÀ CÁC Ô TÍCH CHỌN CỘNG ĐIỂM
if st.session_state["current_results"]:
    st.subheader("📋 Kết quả phân tích & Bộ lọc thêm vào bộ nhớ tích lũy")
    st.info("Hãy tích chọn vào ô bên cạnh câu trả lời bạn muốn lưu vết tiến hóa. Hệ thống sẽ tự động cộng điểm lên đám mây.")
    
    for model_name, ai_response in st.session_state["current_results"].items():
        col_text, col_action = st.columns([5, 1])
        
        with col_text:
            with st.expander(f"📌 {model_name}", expanded=True):
                st.write(ai_response)
                
        with col_action:
            checkbox_key = f"chk_{model_name}_{st.session_state['db_answers_count']}"
            add_to_memory = st.checkbox("Lưu tệp", key=checkbox_key)
            if add_to_memory:
                with st.spinner("Đang lưu..."):
save_single_passed_evolution(user_id, model_name, st.session_state["last_prompt"], ai_response)st.success(f"Đã lưu!")time.sleep(0.4)st.rerun()
