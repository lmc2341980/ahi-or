import streamlit as st
import requests

st.set_page_config(page_title="AHI-V Real Test", layout="wide")
st.title("🧬 AHI-V Real Test (Final Fix)")

# Lấy chìa khóa từ Secrets
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "").strip()

prompt = st.text_input("Nhập câu hỏi để thử bộ não thật (Ấn Enter):")

if prompt:
    # 1. THỬ GEMINI (Khóa AQ.) - CẤU TRÚC URL TÁCH BIỆT
    st.subheader("🤖 Kết quả từ Gemini:")
    try:
        # Đảm bảo URL chính xác và chìa khóa được truyền qua tham số 'key'
        url = "https://googleapis.com"
        params = {"key": GEMINI_KEY}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        r = requests.post(url, params=params, json=payload, timeout=20)
        
        if r.status_code == 200:
            res = r.json()
            # Lấy đúng vị trí dữ liệu trong mảng JSON của Google
            text = res['candidates']['content']['parts']['text']
            st.success(text)
        else:
            st.error(f"Gemini báo lỗi (Mã {r.status_code}): {r.text}")
            
    except Exception as e:
        st.error(f"Lỗi hệ thống Gemini: {e}")

    # 2. THỬ OPENROUTER
    st.subheader("🤖 Kết quả từ OpenRouter:")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": ""content": prompt}]
        }
        
        # OpenRouter yêu cầu URL chuẩn xác cho chat completions
        r = requests.post("https://openrouter.ai", headers=headers, json=data, timeout=20)
        
        if r.status_code == 200:
            res = r.json()
            st.info(res['choices']['message']['content'])
        else:
            st.warning(f"OpenRouter báo lỗi (Mã {r.status_code}): {r.text}")
            
    except Exception as e:
        st.error(f"Lỗi kết nối OpenRouter: {e}")
