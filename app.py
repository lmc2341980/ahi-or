import streamlit as st
import requests
import random

st.set_page_config(page_title="AHI Test", layout="wide")
st.title("🧬 AHI-V Real Test")

# Lấy chìa khóa từ Secrets
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

prompt = st.text_input("Nhập câu hỏi để thử bộ não thật (Ấn Enter):")

if prompt:
    # 1. THỬ GEMINI (Khóa AQ.)
    st.subheader("🤖 Kết quả từ Gemini:")
    try:
        # URL tối giản tuyệt đối để không bao giờ bị lỗi 'Invalid port'
        url = f"https://googleapis.com{GEMINI_KEY}"
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
        res = r.json()
        if 'candidates' in res:
            st.write(res['candidates'][0]['content']['parts'][0]['text'])
        else:
            st.error(f"Gemini báo lỗi: {res}")
    except Exception as e:
        st.error(f"Lỗi kết nối Gemini: {e}")

    # 2. THỬ OPENROUTER (ChatGPT/Claude)
    st.subheader("🤖 Kết quả từ OpenRouter:")
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
        payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
        r = requests.post("https://openrouter.ai", headers=headers, json=payload, timeout=15)
        res = r.json()
        if 'choices' in res:
            st.write(res['choices'][0]['message']['content'])
        else:
            st.error(f"OpenRouter báo lỗi: {res}")
    except Exception as e:
        st.error(f"Lỗi kết nối OpenRouter: {e}")
