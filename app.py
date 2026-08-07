# Đoạn code vá lỗi kết nối cuối cùng
async def fetch_single_ahi_old(ahi_name: str, config: Dict[str, str], prompt: str) -> str:
    provider = config["provider"]
    model = config["model"]
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            if provider == "gemini":
                # Sửa URL: Bỏ dấu gạch chéo dư thừa trước dấu hai chấm
                url = f"https://googleapis.com{model}:generateContent"
                res = await client.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json={"contents": [{"parts": [{"text": prompt}]}]})
                return res.json()['candidates']['content']['parts']['text'] if res.status_code == 200 else f"Lỗi Gemini: {res.status_code}"
            
            elif provider in ["groq", "openrouter"]:
                base = "https://groq.com" if provider == "groq" else "https://openrouter.ai"
                key = GROQ_KEY if provider == "groq" else OPENROUTER_KEY
                res = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "messages": [{"role": ""content": prompt}]})
                if res.status_code == 200:
                    return res.json()['choices']['message']['content']
                return f"Lỗi {provider}: {res.status_code} - Vui lòng nạp thêm tiền vào tài khoản API."
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"
    return "❌ Trục trặc luồng."
