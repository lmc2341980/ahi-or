mport asyncio
import json
import os
import time
import math
from typing import Dict, Any, List, Tuple
import streamlit as st
from openai import AsyncOpenAI
import google.genai as genai

# ==============================================================================
# 1. SETUP & INITIALIZATION
# ==============================================================================
st.set_page_config(page_title="AHI-Orchestrator Engine", layout="wide", page_icon="🧬")

# Khởi tạo thư mục lưu trữ DBRS / DBV mô phỏng
DATA_DIR = "ahi_workspace_db"
os.makedirs(f"{DATA_DIR}/ahi_p", exist_ok=True)
os.makedirs(f"{DATA_DIR}/ahi_old", exist_ok=True)
os.makedirs(f"{DATA_DIR}/ahi_subiet_shared", exist_ok=True)
os.makedirs(f"{DATA_DIR}/ahi_om", exist_ok=True)

# Đọc API Keys từ secrets
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# Khởi tạo Async Clients
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY if OPENROUTER_KEY else "dummy_key"
)

groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY if GROQ_KEY else "dummy_key"
)

# Danh sách 8 AHI-Old Sub-Models được quản lý
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
# 2. HELPER FUNCTIONS FOR DBRS / VECTOR EMBEDDING SIMULATION
# ==============================================================================
def simple_text_vector(text: str) -> List[float]:
    """Mô phỏng Vector Embedding đơn giản từ từ vựng (Bag of Words normalized)."""
    words = text.lower().split()
    vocab = ["ahi", "ai", "tu", "duy", "tien", "hoa", "hien", "phap", "v", "subiet", "hoi", "tra", "loi"]
    vec = [words.count(w) for w in vocab]
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))

def search_ahi_om_similarity(query_text: str) -> Tuple[float, Dict[str, Any]]:
    """So sánh câu hỏi mới với các câu hỏi cũ trong AHI-Om / Subiet DB."""
    q_vec = simple_text_vector(query_text)
    best_sim = 0.0
    best_match = None
    
    # Duyệt file dữ liệu đã lưu trong AHI-Om
    om_file = f"{DATA_DIR}/ahi_om/planet_knowledge.json"
    if os.path.exists(om_file):
        with open(om_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            for rec in records:
                sim = cosine_similarity(q_vec, rec.get("vector", []))
                if sim > best_sim:
                    best_sim = sim
                    best_match = rec
    return best_sim, best_match

# ==============================================================================
# 3. ASYNC DISPATCHER & AHI-V RETRY LOOP
# ==============================================================================
async def fetch_single_ahi_old(ahi_name: str, config: Dict[str, str], prompt: str) -> str:
    """Gọi API nguyên bản từ AHI-Old."""
    try:
        provider = config["provider"]
        model = config["model"]

        if provider == "openrouter":
            res = await openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content
        elif provider == "groq":
            res = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content
        elif provider == "gemini":
            if not GEMINI_KEY:
                return f"[{ahi_name}] GEMINI_API_KEY chưa được cấu hình."
            client = genai.Client(api_key=GEMINI_KEY)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text
        return f"[{ahi_name}] Provider không hợp lệ."
    except Exception as e:
        return f"[{ahi_name} Error]: {str(e)}"

async def fetch_and_validate_ahi(ahi_name: str, config: Dict[str, str], prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    """Luồng gọi AI + AHI-V Kiểm soát Hiến pháp (Có vòng lặp Retry nếu vi phạm)."""
    current_prompt = prompt
    attempts = 0
    
    while attempts <= max_retries:
        attempts += 1
        raw_response = await fetch_single_ahi_old(ahi_name, config, current_prompt)
        
        # AHI-V Engine Check (Kiểm soát Ràng buộc & Hiến pháp AHI)
        ahi_v_passed = True
        violation_reason = ""
        
        # Ví dụ Rule Engine AHI-V: Không chứa từ phản cảm, độ dài > 20 ký tự, không bị lỗi API
        if "Error" in raw_response or len(raw_response.strip()) < 10:
            ahi_v_passed = False
            violation_reason = "Câu trả lời quá ngắn hoặc gặp lỗi API."
        elif "KHÔNG_TUAN_THU_HIEN_PHAP" in raw_response:
            ahi_v_passed = False
            violation_reason = "Vi phạm điều khoản Hiến pháp AHI."

        if ahi_v_passed:
            # Ghi nhận kết quả hợp lệ vào DBRS / DBV của AHI-Old tương ứng
            save_record = {
                "timestamp": time.time(),
                "prompt": prompt,
                "response": raw_response,
                "attempts": attempts,
                "ahi_v_status": "PASSED"
            }
            db_path = f"{DATA_DIR}/ahi_old/{ahi_name}.json"
            history = []
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            history.append(save_record)
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            return {
                "ahi_name": ahi_name,
                "response": raw_response,
                "ahi_v_status": "PASSED",
                "attempts": attempts
            }
        else:
            # Gửi lại câu hỏi kèm Feedback điều chỉnh cho AHI-Or đưa sang AHI-Old tương ứng
            current_prompt = f"{prompt}\n\n[HỆ THỐNG AHI-V YÊU CẦU ĐIỀU CHỈNH - LẦN {attempts}]: Câu trả lời trước vi phạm: {violation_reason}. Hãy trả lời lại đầy đủ và tuân thủ chuẩn mực."

    return {
        "ahi_name": ahi_name,
        "response": f"Thất bại sau {max_retries} lần kiểm duyệt AHI-V.",
        "ahi_v_status": "FAILED",
        "attempts": attempts
    }

async def dispatch_all_ahi_models(prompt: str) -> List[Dict[str, Any]]:
    """Bắn câu hỏi đồng thời đến 8 AHI-Old qua asyncio.gather()."""
    tasks = [
        fetch_and_validate_ahi(ahi_name, config, prompt)
        for ahi_name, config in AHI_OLD_MODELS.items()
    ]
    return await asyncio.gather(*tasks)

# ==============================================================================
# 4. STREAMLIT UI & WORKSPACE ENGINE
# ==============================================================================
st.title("🧬 AHI-Orchestrator Workspace & Multi-AI Dispatcher")
st.caption("Hệ sinh thái Tri thức Tiến hóa - Điều phối & Đánh giá Đa mô hình AHI")

# Sidebar - Định danh người dùng AHI-P
st.sidebar.header("👤 Định Danh Người Dùng (AHI-P)")
user_id = st.sidebar.text_input("Mã định danh AHI-P:", value="AHI-P-EXPERT-01")
p_profile = st.sidebar.selectbox("Lĩnh vực chuyên gia:", ["AI Software Architect", "Data Scientist", "System Engineer", "Philosophy & Ethics"])

# Nơi lưu trữ thông tin & ký ức của AHI-P
p_file_path = f"{DATA_DIR}/ahi_p/{user_id}.json"
ahi_p_data = {"identity": user_id, "domain": p_profile, "accepted_knowledge": [], "workspace_chats": []}
if os.path.exists(p_file_path):
    with open(p_file_path, "r", encoding="utf-8") as f:
        ahi_p_data = json.load(f)

st.sidebar.success(f"Đã kết nối AHI-P: **{user_id}**")
st.sidebar.info(f"Tổng câu trả lời đã tích lũy tiến hóa: **{len(ahi_p_data['accepted_knowledge'])}**")

# Main Interface: AHI-Workspace Session
st.subheader("💬 AHI-Workspace (Phiên Làm Việc Không Giới Hạn Space/Time)")

# Hiển thị lịch sử workspace
for chat in ahi_p_data.get("workspace_chats", []):
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_query = st.chat_input("Nhập câu hỏi hoặc yêu cầu điều phối AHI...")

if user_query:
    # 1. Ghi nhận câu hỏi vào UI & Workspace AHI-P
    ahi_p_data["workspace_chats"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Kiểm tra trùng lặp & So sánh chất lượng với AHI-Om / Shared DB
    sim_score, best_match = search_ahi_om_similarity(user_query)
    if sim_score > 0.8:
        st.warning(f"🔍 **AHI-Or Alert:** Phát hiện câu hỏi trùng lặp cao ({sim_score*100:.1f}%) với Tri thức AHI-Om cũ!")
        st.info(f"**Câu hỏi cũ trong DB:** {best_match.get('prompt')}\n\n**Chất lượng AHI-SuBiet đánh giá:** {best_match.get('score')} điểm.")

    # 3. Tiến hành gọi đồng thời 8 AHI-Old qua AHI-Or Dispatcher
    with st.spinner("🚀 AHI-Or đang gửi câu hỏi tới 8 AHI-Old và kiểm duyệt qua AHI-V..."):
        results = asyncio.run(dispatch_all_ahi_models(user_query))

    # 4. Hiển thị kết quả thu được từ các AHI-Old tương ứng
    st.write("### 🤖 Kết Quả Thu Được Từ Các AHI-Old (Đã Qua AHI-V):")
    cols = st.columns(2)
    
    valid_candidates = []
    for idx, res in enumerate(results):
        col = cols[idx % 2]
        with col:
            with st.expander(f"📌 {res['ahi_name']} | Status: {res['ahi_v_status']} ({res['attempts']} attempts)", expanded=True):
                st.write(res["response"])
                if res["ahi_v_status"] == "PASSED":
                    valid_candidates.append(res)

    # 5. AHI-SuBiet Đánh Giá & Chọn Lựa
    st.write("---")
    st.subheader("⚖️ AHI-SuBiet Evaluator & Đề Xuất Tiến Hóa")
    
    if valid_candidates:
        # Giả lập AHI-SuBiet chọn ra câu trả lời xuất sắc nhất
        best_candidate = max(valid_candidates, key=lambda x: len(x["response"])) # Mẫu đánh giá
        
        st.success(f"🏆 **AHI-SuBiet đề xuất câu trả lời từ [{best_candidate['ahi_name']}]** là tối ưu nhất cho AHI-P!")
        st.markdown(best_candidate["response"])
        
        # Người dùng AHI-P lựa chọn chấp nhận câu trả lời
        col_acc, col_rej = st.columns(2)
        if col_acc.button("✅ Người dùng AHI-P Chấp Nhận (Đưa Vào Tiến Hóa Cá Nhân)"):
            # A. Đưa vào Tri thức Tiến hóa cá nhân của AHI-P
            accepted_record = {
                "timestamp": time.time(),
                "query": user_query,
                "response": best_candidate["response"],
                "source_ahi": best_candidate["ahi_name"]
            }
            ahi_p_data["accepted_knowledge"].append(accepted_record)
            ahi_p_data["workspace_chats"].append({"role": "assistant", "content": f"**[{best_candidate['ahi_name']} Accepted]:** {best_candidate['response']}"})
            
            with open(p_file_path, "w", encoding="utf-8") as f:
                json.dump(ahi_p_data, f, ensure_ascii=False, indent=2)
                
            # B. Đưa vào Kho Shared Preview của AHI-SuBiet (Cho AHI-P khác xem tham khảo)
            shared_record = {
                "id": f"PREVIEW-{int(time.time())}",
                "prompt": user_query,
                "response": best_candidate["response"],
                "author_ahi_p": user_id,
                "source": best_candidate["ahi_name"],
                "status": "PREVIEW_ONLY", # Chưa phải DBRS/DBV chính thức của AHI-Om
                "vector": simple_text_vector(user_query)
            }
            shared_file = f"{DATA_DIR}/ahi_subiet_shared/preview_db.json"
            shared_list = []
            if os.path.exists(shared_file):
                with open(shared_file, "r", encoding="utf-8") as f:
                    shared_list = json.load(f)
            shared_list.append(shared_record)
            with open(shared_file, "w", encoding="utf-8") as f:
                json.dump(shared_list, f, ensure_ascii=False, indent=2)
                
            st.balloons()
            st.success("🎉 Đã lưu thành công vào **Tri thức Tiến hóa của AHI-P** và cập nhật vào **Kho Tham Khảo AHI-SuBiet**!")
    else:
        st.error("❌ Không có câu trả lời nào vượt qua được tiêu chuẩn AHI-V.")
