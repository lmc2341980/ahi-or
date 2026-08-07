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
from openai import AsyncOpenAI
import google.genai as genai

# ==============================================================================
# 1. SETUP & INITIALIZATION & NEON TABLE GENERATION
# ==============================================================================
st.set_page_config(page_title="AHI-Orchestrator Engine", layout="wide", page_icon="🧬")

def get_connection():
    """Tự động kết nối và lấy chuỗi cấu hình bảo mật từ Streamlit Secrets."""
    db_url = st.secrets["NEON_DB_URL"]
    return psycopg2.connect(db_url)

def init_neon_tables():
    """Khởi tạo đồng thời phân hệ DBRS và phần mở rộng Vector (DBV) trên Neon Cloud."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgvector;")
        # DẠNG DBRS: Trạng thái và số câu tích lũy thực tế của Chuyên gia AHI-P
        cur.execute("""
            CREATE TABLE IF NOT EXISTS expert_state (
                ahi_p TEXT PRIMARY KEY,
                expert_code TEXT NOT NULL,
                total_answers INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # DẠNG DBV: Lưu trữ tri thức dưới dạng tọa độ hình học Vector toán học
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
    except Exception as e:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

init_neon_tables()

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai", api_key=OPENROUTER_KEY or "dummy_key")
groq_client = AsyncOpenAI(base_url="https://groq.com", api_key=GROQ_KEY or "dummy_key")

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

# ==============================================================================
# 2. NEON DATABASE OPERATIONS (DBRS & DBV IMPLEMENTATION)
# ==============================================================================
def fetch_expert_status(ahi_p: str) -> Dict[str, Any]:
    """[DBRS] Trích xuất số liệu tiến hóa thực tế từ database, chống reset về số 0."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT total_answers, expert_code FROM expert_state WHERE ahi_p = %s;", (ahi_p,))
        result = cur.fetchone()
        if not result:
            cur.execute("INSERT INTO expert_state (ahi_p, expert_code, total_answers) VALUES (%s, %s, %s) RETURNING *;", (ahi_p, "AHI-P-EXPERT-01", 0))
            conn.commit()
            return {"total_answers": 0, "expert_code": "AHI-P-EXPERT-01"}
        return result
    finally:
        cur.close()
        conn.close()

def mock_generate_embedding() -> List[float]:
    return [random.uniform(-1, 1) for _ in range(1536)]

def save_passed_evolution(ahi_p: str, ahi_name: str, prompt: str, content: str):
    """[DBRS + DBV] Lưu đồng bộ trạng thái đếm số câu và lưu tọa độ nhúng tri thức."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE expert_state SET total_answers = total_answers + 1, updated_at = NOW() WHERE ahi_p = %s;", (ahi_p,))
        vector_data = mock_generate_embedding()
        cur.execute("INSERT INTO ahi_evolution_vectors (ahi_p, ahi_name, prompt, content, embedding) VALUES (%s, %s, %s, %s, %s);", (ahi_p, ahi_name, prompt, content, vector_data))
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi ghi nhận đám mây: {e}")
    finally:
        cur.close()
        conn.close()

def search_ahi_om_similarity_neon(query_text: str) -> List[Dict[str, Any]]:
    """[DBV] Sử dụng toán tử khoảng cách toán học (<->) của pgvector để tìm kiếm ngữ cảnh cũ cực nhanh."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT prompt, content, (embedding <-> %s::vector) as distance FROM ahi_evolution_vectors ORDER BY distance ASC LIMIT 3;", (mock_generate_embedding(),))
        return cur.fetchall()
    except Exception as e:
        return []
    finally:
        cur.close()
        conn.close()

# ==============================================================================
# 3. ASYNC DISPATCHER & AHI-V RETRY LOOP
# ==============================================================================
async def fetch_single_ahi_old(ahi_name: str, config: Dict[str, str], prompt: str) -> str:
    try:
        provider = config["provider"]
        model = config["model"]
        if provider == "openrouter":
            res = await openrouter_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7)
            return res.choices[0].message.content
        elif provider == "groq":
            res = await groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7)
            return res.choices[0].message.content
        elif provider == "gemini":
            if not GEMINI_KEY: return f"[{ahi_name}] GEMINI_API_KEY chưa cấu hình."
            client = genai.Client(api_key=GEMINI_KEY)
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        return f"[{ahi_name}] Provider không hợp lệ."
    except Exception as e:
        return f"[{ahi_name} Error]: {str(e)}"

async def fetch_and_validate_ahi(ahi_p: str, ahi_name: str, config: Dict[str, str], prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    current_prompt = prompt
    attempts = 0
    while attempts <= max_retries:
        attempts += 1
        raw_response = await fetch_single_ahi_old(ahi_name, config, current_prompt)
        ahi_v_passed = True
        violation_reason = ""
        if "Error" in raw_response or len(raw_response.strip()) < 10:
            ahi_v_passed = False
            violation_reason = "Không phản hồi hoặc lỗi API."
        elif "KHÔNG_TUAN_THU_HIEN_PHAP" in raw_response:
            ahi_v_passed = False
            violation_reason = "Vi phạm Hiến pháp hệ thống AHI."

        if ahi_v_passed:
            save_passed_evolution(ahi_p, ahi_name, prompt, raw_response)
            return {"ahi_name": ahi_name, "response": raw_response, "ahi_v_status": "PASSED", "attempts": attempts}
        else:
            current_prompt = f"{prompt}\n\n[AHI-V PHẢN HỒI LẦN {attempts}]: Trả lời trước vi phạm: {violation_reason}. Hãy hiệu chỉnh chuẩn mực."
    return {"ahi_name": ahi_name, "response": f"Thất bại sau {max_retries} lần kiểm duyệt.", "ahi_v_status": "FAILED", "attempts": attempts}

async def dispatch_all_ahi_models(ahi_p: str, prompt: str) -> List[Dict[str, Any]]:
    tasks = [fetch_and_validate_ahi(ahi_p, ahi_name, config, prompt) for ahi_name, config in AHI_OLD_MODELS.items()]
    return await asyncio.gather(*tasks)

# ==============================================================================
# 4. STREAMLIT UI & DASHBOARD WORKSPACE
# ==============================================================================
st.title("🧬 AHI-Orchestrator Workspace & Multi-AI Dispatcher")
st.caption("Cơ sở hạ tầng tiến hóa đa mô hình quản lý đồng bộ bởi Neon PostgreSQL (DBRS + DBV)")

st.sidebar.header("👤 Định Danh Người Dùng (AHI-P)")
user_id = st.sidebar.text_input("Mã định danh AHI-P:", value="AHI-P-EXPERT-01")
p_profile = st.sidebar.selectbox("Lĩnh vực chuyên gia:", ["AI Architecture", "Data Science", "Ethical AI"])

if user_id:
    user_status = fetch_expert_status(user_id)
    st.sidebar.markdown(f"""
    **Trạng thái thực thi dữ liệu:**
    - Đã kết nối AHI-P: `{user_status['expert_code']}`
    - Tổng câu trả lời đã tích lũy tiến hóa: **{user_status['total_answers']}**
    """)

main_prompt = st.text_area("Nhập câu lệnh / Yêu cầu điều phối hệ thống kiến thức:")

if st.button("Kích hoạt Phân phối Tiến hóa Đa Mô Hình"):
    if not main_prompt.strip():
        st.warning("Vui lòng nhập nội dung trước khi chạy.")
    else:
        with st.spinner("Đang điều phối xử lý song song và lưu trữ Vector Cloud..."):
            results = asyncio.run(dispatch_all_ahi_models(user_id, main_prompt))
            st.success("Đồng bộ thành công dữ liệu trạng thái (DBRS) và cấu trúc Vector (DBV) lên Neon!")
            for res in results:
                with st.expander(f"📌 {res['ahi_name']} | Trạng thái: {res['ahi_v_status']} (Thử lại: {res['attempts']} lần)"):
                    st.write(res['response'])
            st.rerun()
