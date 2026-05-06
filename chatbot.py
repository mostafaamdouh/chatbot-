from flask import Flask, app, request, jsonify, render_template_string
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path
import re

# LOAD ENV 

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(f"GROQ_API_KEY not found. Check .env at: {env_path}")

client = Groq(api_key=api_key)


# AGENCY DATA (EN + AR)

AGENCY_DATA_EN = """
You are a helpful customer service assistant for a professional advertising agency and photo studio.

COMPANY INFORMATION:
- We are a full-service advertising agency and creative studio
- We specialize in high-quality photo sessions and creative campaigns

WORKING HOURS:
- Open: Saturday to Thursday, 9 AM to 6 PM
- Closed: Every Friday
- Location: [Your City - update this]

SERVICES WE OFFER:
1) PHOTOGRAPHY: product, portraits/headshots, fashion/lifestyle, food, real estate, events
2) VIDEO PRODUCTION: commercials, social media content, corporate, promotional videos
3) ADVERTISING & MARKETING: social ads, Google Ads, brand strategy, logo/identity
4) CREATIVE DESIGN: print/digital design, social content, brochures, packaging
5) SOCIAL MEDIA MANAGEMENT: content planning, community management, influencers

PRICING (Examples - update):
- Basic photo session: Starting at $200
- Product photography: Starting at $150 per product
- Social media package: Starting at $500/month
- Custom video production: Starting at $1,000

BOOKING:
Collect: name, service needed, preferred date/time, phone/email.
Confirm: "We'll contact you within 24 hours to confirm."
Be friendly, professional, concise, and helpful.
"""

AGENCY_DATA_AR = """
أنت مساعد خدمة عملاء محترف لوكالة دعاية وإعلان واستوديو تصوير.

معلومات الشركة:
- وكالة دعاية وإعلان متكاملة + استوديو إبداعي
- متخصصون في جلسات تصوير عالية الجودة وحملات إبداعية

مواعيد العمل:
- مفتوح: من السبت إلى الخميس، 9 صباحًا إلى 6 مساءً
- مغلق: كل يوم جمعة
- الموقع: [اكتب المدينة هنا]

الخدمات:
1) التصوير: تصوير منتجات، بورتريه/هيدشوت، أزياء ولايف ستايل، طعام ومشروبات، عقارات، تغطية فعاليات
2) الفيديو: إعلانات، محتوى سوشيال، فيديوهات شركات، فيديوهات ترويجية
3) التسويق والإعلانات: حملات سوشيال، جوجل أدز، استراتيجية علامة تجارية، تصميم لوجو وهوية
4) التصميم الإبداعي: تصميم مطبوعات ورقمي، محتوى سوشيال، بروشورات، تصميم باكدچ
5) إدارة السوشيال: خطة محتوى، جدولة، إدارة مجتمع، تعاون مع إنفلونسرز

الأسعار (أمثلة - عدّلها):
- جلسة تصوير بسيطة: تبدأ من 200$
- تصوير منتجات: يبدأ من 150$ للمنتج
- باكدج سوشيال: يبدأ من 500$/شهريًا
- فيديو مخصص: يبدأ من 1000$

الحجز:
اجمع: الاسم، الخدمة المطلوبة، التاريخ/الوقت المناسب، رقم الهاتف/الإيميل.
أكد: "هنكلمك خلال 24 ساعة لتأكيد الحجز."
خليك لطيف، محترف، مختصر، ومفيد.
"""

# Language detection

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def detect_lang(text: str) -> str:
    """Return 'ar' if Arabic letters are present, else 'en'."""
    return "ar" if ARABIC_RE.search(text) else "en"

def system_prompt_for(lang: str) -> str:
    if lang == "ar":
        return (
            AGENCY_DATA_AR
            + "\n\nالتزم بالرد باللغة العربية فقط عندما يكتب العميل بالعربية. "
              "لو سأل بالإنجليزية، رد بالإنجليزية فقط."
        )
    return (
        AGENCY_DATA_EN
        + "\n\nReply in English only when the customer writes in English. "
          "If the customer writes in Arabic, reply in Arabic only."
    )


# Chat history per customer

conversation_history = []

def chat(user_message: str) -> str:
    global conversation_history

    lang = detect_lang(user_message)

    # Ensure system message is correct for this turn's language
    system_msg = {"role": "system", "content": system_prompt_for(lang)}

    # Keep history but always prepend system message matching the current language
    messages = [system_msg] + conversation_history + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=900,
    )

    bot_response = response.choices[0].message.content

    # Save user+assistant messages 
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": bot_response})

    return bot_response

def reset_customer():
    global conversation_history
    conversation_history = []

print("Agency chatbot initialized!")
print("Ask about services, pricing, or booking!")
print("Type 'quit' to exit. Type 'new' for a new customer.\n")

while True:
    user_input = input("Customer: ").strip()

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("\nBot: Thank you for contacting us! Have a great day!")
        break

    if user_input.lower() in ["new", "reset"]:
        reset_customer()
        print(" Ready for a new customer!\n")
        continue

    if not user_input:
        continue

    try:
        reply = chat(user_input)
        print(f"Bot: {reply}\n")
        print("-" * 60)
    except Exception as e:
        print(f" Error: {e}")
        break

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)