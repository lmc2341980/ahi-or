import streamlit as st
import requests

st.set_page_config(page_title="AHI-V Final Test", layout="wide")
st.title("🧬 AHI-V Real Test (Fix URL)")

# Lấy chìa khóa từ Secrets
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

prompt = st.text_input("Nhập câu hỏi (Ấn Enter):")

if prompt:
    # 1. THỬ GEMINI (Khóa AQ.) - SỬA LỖI URL DÍNH CHỮ
    st.subheader("🤖 Kết quả từ Gemini:")
    try:
        # Cấu hình URL chuẩn xác, có đầy đủ phân tầng v1beta
        url = f"https://googleapis.com{GEMINI_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        r = requests.post(url, json=payload, timeout=20)
        res = r.json()
        
        if 'candidates' in res:
            # Lấy đúng vị trí dữ liệu trong mảng JSON
            text = res['candidates'][0]['content']['parts'][0]['text']
            st.success(text)
        else:
            st.error(f"Gemini báo lỗi: {res}")
    except Exception as e:
        st.error(f"Lỗi hệ thống Gemini: {e}")

    # 2. THỬ OPENROUTER (Nếu hết tiền sẽ báo rõ)
    st.subheader("🤖 Kết quả từ OpenRouter:")
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
        data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
        
        r = requests.post("https://openrouter.ai", headers=headers, json=data, timeout=20)
        res = r.json()
        
        if 'choices' in res:
            st.info(res['choices'][0]['message']['content'])
        else:
            st.warning(f"OpenRouter báo lỗi (Thường do hết tiền): {res}")
    except Exception as e:
        st.error(f"Lỗi kết nối OpenRouter: {e}")
