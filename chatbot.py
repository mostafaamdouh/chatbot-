from flask import Flask, request, jsonify
from groq import Groq
import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path
import re
from datetime import datetime, timedelta
from collections import deque

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found.")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY not found.")

groq_client = Groq(api_key=groq_api_key)
app = Flask(__name__)

from genz_data import GENZ_STUDIOS_EN, GENZ_STUDIOS_AR
from rag import search
from booking_tools import TOOLS, execute_tool

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SESSION_TTL_DAYS   = 3
MAX_MESSAGE_LENGTH = 500
RATE_LIMIT         = 20

# ─────────────────────────────────────────────────────────────────────────────
#  SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
sessions = {}


# ─────────────────────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
ARABIC_RE  = re.compile(r"[\u0600-\u06FF]+")
ENGLISH_RE = re.compile(r"[a-zA-Z]+")

def detect_lang(text: str) -> str:
    arabic_chars  = sum(len(m) for m in ARABIC_RE.findall(text))
    english_chars = sum(len(m) for m in ENGLISH_RE.findall(text))
    total = arabic_chars + english_chars
    if total == 0:
        return "ar"
    return "ar" if (arabic_chars / total) >= 0.5 else "en"


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_expired_sessions():
    cutoff  = datetime.now() - timedelta(days=SESSION_TTL_DAYS)
    expired = [sid for sid, data in sessions.items() if data["last_active"] < cutoff]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"[Session Cleanup] Removed {len(expired)} expired session(s).")


def is_rate_limited(session_id: str) -> bool:
    now          = datetime.now()
    window_start = now - timedelta(seconds=60)
    timestamps   = sessions[session_id]["timestamps"]
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT:
        return True
    timestamps.append(now)
    return False


def validate_message(message: str, lang: str) -> str | None:
    if not message:
        return "الرسالة فارغة." if lang == "ar" else "Message is empty."
    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            f"الرسالة طويلة جداً. الحد الأقصى {MAX_MESSAGE_LENGTH} حرف."
            if lang == "ar"
            else f"Message too long. Maximum is {MAX_MESSAGE_LENGTH} characters."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
BOOKING_REDIRECT_EN = """
BOOKING INSTRUCTIONS:
- You can check studio availability when the client asks about free slots or timings.
- You CANNOT create or cancel bookings in this chat.
- If the client wants to make a booking, tell them:
  "To book a studio, please use the GENZ Studios app or call us:
   New Cairo branch: 010-0000-0001 | Dokki branch: 010-0000-0002"
- Today's date is: {date} ({day}).
"""

BOOKING_REDIRECT_AR = """
تعليمات الحجز:
- تقدر تتحقق من مواعيد الاستوديوهات الفاضية لما العميل يسأل.
- مش بتعمل حجوزات أو إلغاءات مباشرة في الشات ده.
- لو العميل عايز يحجز، قوله:
  "للحجز، استخدم تطبيق GENZ Studios أو اتصل بينا:
   فرع التجمع: 010-0000-0001 | فرع الدقي: 010-0000-0002"
- تاريخ النهارده: {date} ({day}).
"""

def system_prompt_for(lang: str, user_message: str) -> str:
    context = search(user_message)
    today   = datetime.now().strftime("%Y-%m-%d")
    day     = datetime.now().strftime("%A")

    if lang == "ar":
        return (
            GENZ_STUDIOS_AR
            + BOOKING_REDIRECT_AR.format(date=today, day=day)
            + f"\n\nمعلومات ذات صلة بسؤال العميل:\n{context}"
        )
    return (
        GENZ_STUDIOS_EN
        + BOOKING_REDIRECT_EN.format(date=today, day=day)
        + f"\n\nRelevant information for this question:\n{context}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION CALLING LOOP
#  Handles check_availability and get_booking tool calls from the LLM.
#  Max 5 iterations to prevent infinite loops.
# ─────────────────────────────────────────────────────────────────────────────
def call_groq_with_tools(messages: list, session_id: str, lang: str) -> str:
    loop_messages = list(messages)

    for _ in range(5):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=loop_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=900,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        loop_messages.append({
            "role":       "assistant",
            "content":    message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            print(f"[Tool Call] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args, session_id)
            print(f"[Tool Result] {result}")
            loop_messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return (
        "عذراً، في مشكلة في معالجة طلبك. حاول تاني."
        if lang == "ar"
        else "Sorry, there was an issue processing your request. Please try again."
    )


def call_openrouter_with_tools(messages: list, session_id: str, lang: str) -> str:
    loop_messages = list(messages)

    for _ in range(5):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model":       "meta-llama/llama-3.3-70b-instruct:free",
                "messages":    loop_messages,
                "tools":       TOOLS,
                "tool_choice": "auto",
                "temperature": 0.7,
                "max_tokens":  900,
            },
            timeout=30,
        )
        response.raise_for_status()
        data    = response.json()
        message = data["choices"][0]["message"]

        if not message.get("tool_calls"):
            return message["content"]

        loop_messages.append({
            "role":       "assistant",
            "content":    message.get("content") or "",
            "tool_calls": message["tool_calls"],
        })

        for tc in message["tool_calls"]:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])
            print(f"[Tool Call - OR] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args, session_id)
            print(f"[Tool Result - OR] {result}")
            loop_messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result,
            })

    return (
        "عذراً، في مشكلة في معالجة طلبك. حاول تاني."
        if lang == "ar"
        else "Sorry, there was an issue processing your request. Please try again."
    )


def get_ai_reply(messages: list, session_id: str, lang: str) -> str:
    try:
        return call_groq_with_tools(messages, session_id, lang)
    except Exception as e:
        print(f"[Groq failed] {e} — switching to OpenRouter...")
    try:
        return call_openrouter_with_tools(messages, session_id, lang)
    except Exception as e:
        print(f"[OpenRouter failed] {e}")
    return (
        "عذراً، في مشكلة تقنية دلوقتي. حاول تاني بعد شوية."
        if lang == "ar"
        else "Sorry, we're experiencing a technical issue. Please try again in a moment."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "GENZ Chatbot Server Running"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        cleanup_expired_sessions()

        data         = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        session_id   = data.get("session_id", "default")

        if not user_message:
            return jsonify({"reply": ""}), 400

        if session_id not in sessions:
            sessions[session_id] = {
                "history":     [],
                "last_active": datetime.now(),
                "timestamps":  deque(),
            }

        sessions[session_id]["last_active"] = datetime.now()
        history = sessions[session_id]["history"]

        lang = detect_lang(user_message)

        error = validate_message(user_message, lang)
        if error:
            return jsonify({"reply": error}), 400

        if is_rate_limited(session_id):
            msg = (
                "بتبعت رسايل كتير أوي! استنى دقيقة وحاول تاني."
                if lang == "ar"
                else "Too many messages! Please wait a moment and try again."
            )
            return jsonify({"reply": msg}), 429

        messages = (
            [{"role": "system", "content": system_prompt_for(lang, user_message)}]
            + history
            + [{"role": "user", "content": user_message}]
        )

        bot_reply = get_ai_reply(messages, session_id, lang)

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": bot_reply})

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"[Chat Error] {e}")
        lang = detect_lang(data.get("message", "") if "data" in dir() else "")
        msg  = (
            "عذراً، في مشكلة تقنية. حاول تاني بعد شوية."
            if lang == "ar"
            else "Sorry, a technical issue occurred. Please try again in a moment."
        )
        return jsonify({"reply": msg}), 500


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    sessions.pop(session_id, None)
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
