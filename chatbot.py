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

#  CONSTANTS 
SESSION_TTL_DAYS   = 3    # Sessions expire after 3 days of inactivity
MAX_MESSAGE_LENGTH = 500  # Maximum message length
RATE_LIMIT         = 20   # Maximum messages per minute per session

#  SESSIONS 
# Each session stores:
#   history:     Complete conversation history (user + assistant + tool messages)
#   last_active: Last time the client was active (for TTL)
#   timestamps:  Deque with timestamps from the last 60 seconds (for rate limiting)
sessions = {}


#  LANGUAGE DETECTION 
ARABIC_RE  = re.compile(r"[\u0600-\u06FF]+")  # Regex for Arabic characters
ENGLISH_RE = re.compile(r"[a-zA-Z]+")  # Regex for English characters

def detect_lang(text: str) -> str:
    """
    Determines the language of the message by percentage, not just by presence.
    If 50%+ of characters are Arabic → ar, otherwise → en.
    Numbers and symbols are ignored (they have no language).
    """
    arabic_chars  = sum(len(m) for m in ARABIC_RE.findall(text))
    english_chars = sum(len(m) for m in ENGLISH_RE.findall(text))
    total = arabic_chars + english_chars
    if total == 0:
        return "ar"  # default لو مفيش حروف (أرقام فقط مثلاً)
    return "ar" if (arabic_chars / total) >= 0.5 else "en"


#  SESSION HELPERS 
def cleanup_expired_sessions():
    """
    Deletes sessions that haven't been active for more than SESSION_TTL_DAYS.
    Runs automatically with each request.
    """
    cutoff  = datetime.now() - timedelta(days=SESSION_TTL_DAYS)
    expired = [sid for sid, data in sessions.items() if data["last_active"] < cutoff]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"[Session Cleanup] Removed {len(expired)} expired session(s).")


def is_rate_limited(session_id: str) -> bool:
    """
    Sliding window rate limiter.
    Removes timestamps that are outside the 60-second window.
    If message count >= RATE_LIMIT → blocked.
    """
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
    """Validates the message. Returns error string if there's a problem, otherwise None."""
    if not message:
        return "الرسالة فارغة." if lang == "ar" else "Message is empty."
    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            f"الرسالة طويلة جداً. الحد الأقصى {MAX_MESSAGE_LENGTH} حرف."
            if lang == "ar"
            else f"Message too long. Maximum is {MAX_MESSAGE_LENGTH} characters."
        )
    return None


#  SYSTEM PROMPT 
def system_prompt_for(lang: str, user_message: str) -> str:
    """
    Builds the system prompt from:
    1. Complete company information (genz_data.py)
    2. The most relevant chunks for the question (RAG from rag.py)
    3. Booking instructions and tools (new)
    """
    context = search(user_message)

    # Additional booking instructions — explains to the LLM how to handle the tools
    booking_instructions_en = """
BOOKING ASSISTANT INSTRUCTIONS:
- You have access to 4 booking tools: check_availability, create_booking, get_booking, cancel_booking.
- When a client wants to book: first check availability, show options, collect name + phone, then confirm before creating.
- Today's date is: """ + datetime.now().strftime("%Y-%m-%d") + """ (""" + datetime.now().strftime("%A") + """).
- Always collect missing info step by step — don't ask for everything at once.
- After a successful booking, clearly show the booking ID, date, time, studio, and total price.
- Prices are in Egyptian Pounds (EGP).
"""

    booking_instructions_ar = """
تعليمات نظام الحجز:
- عندك 4 أدوات للحجز: check_availability, create_booking, get_booking, cancel_booking.
- لما العميل يطلب حجز: اتحقق من المواعيد الأول، وريه الخيارات، اجمع الاسم والتليفون، تأكد منه، وبعدين عمل الحجز.
- تاريخ النهارده: """ + datetime.now().strftime("%Y-%m-%d") + """ (""" + datetime.now().strftime("%A") + """).
- اجمع المعلومات الناقصة خطوة خطوة — متسألش كل حاجة مرة واحدة.
- بعد الحجز، وضّح رقم الحجز والتاريخ والوقت والاستوديو والسعر الكلي.
- الأسعار بالجنيه المصري.
"""

    if lang == "ar":
        return (
            GENZ_STUDIOS_AR
            + booking_instructions_ar
            + f"\n\nمعلومات ذات صلة بسؤال العميل:\n{context}"
        )
    return (
        GENZ_STUDIOS_EN
        + booking_instructions_en
        + f"\n\nRelevant information for this question:\n{context}"
    )


#  FUNCTION CALLING LOOP 
def call_groq_with_tools(messages: list, session_id: str, lang: str) -> str:
    """
    Sends a request to Groq with tools and handles function calling.

    Flow:
    1. Sends messages + tools to the LLM
    2. If the LLM decides to call a tool:
       a. Execute the actual function
       b. Add the result to messages as a tool message
       c. Send back to the LLM to continue the response
    3. If the LLM replies with regular text → return it

    Why a loop instead of a single call?
    Because the LLM can call multiple tools in the same conversation.
    Example: check_availability → client agrees → create_booking
    """
    loop_messages = list(messages)  # Copy to avoid modifying the original

    for _ in range(5):  # Max 5 tool calls to prevent infinite loop
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=loop_messages,
            tools=TOOLS,
            tool_choice="auto",  # Let the LLM decide when to call tools
            temperature=0.7,
            max_tokens=900,
        )

        message = response.choices[0].message

        # If no tool call → the LLM replied with regular text, done
        if not message.tool_calls:
            return message.content

        #  There are one or more tool calls 
        # Add the LLM response (which contains tool_calls) to messages
        loop_messages.append({
            "role": "assistant",
            "content": message.content or "",
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

        # Execute each tool and add its result
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
        # The loop will repeat and send to the LLM again with tool results

    # لو وصلنا لـ 5 loops من غير رد نهائي
    return (
        "عذراً، في مشكلة في معالجة طلبك. حاول تاني."
        if lang == "ar"
        else "Sorry, there was an issue processing your request. Please try again."
    )


def call_openrouter_with_tools(messages: list, session_id: str, lang: str) -> str:
    """
    Same logic exactly, but with OpenRouter as fallback.
    OpenRouter supports the same tools format.
    """
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
    """Try Groq first, if it fails try OpenRouter, if both fail return error message."""
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


#  FLASK ROUTES 
@app.route("/")
def home():
    return "GENZ Chatbot Server Running"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        cleanup_expired_sessions()

        data          = request.get_json(force=True)
        user_message  = data.get("message", "").strip()
        session_id    = data.get("session_id", "default")

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

        # Input validation
        error = validate_message(user_message, lang)
        if error:
            return jsonify({"reply": error}), 400

        # Rate limiting
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

        # Save only user + assistant in history
        # We don't save tool messages because the system prompt is rebuilt from scratch each time
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


#  ADMIN ENDPOINTS 
# Additional endpoints for admin to view bookings
from database import get_all_bookings

@app.route("/bookings", methods=["GET"])
def bookings():
    """Returns all bookings. Can filter with ?status=confirmed or cancelled"""
    status = request.args.get("status")
    return jsonify(get_all_bookings(status=status))


@app.route("/availability", methods=["GET"])
def availability():
    """
    Checks available times for a studio on a given day.
    Example: GET /availability?studio=A&date=2026-06-09&duration=2
    """
    from database import check_availability as db_check
    studio   = request.args.get("studio", "A")
    date     = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    duration = int(request.args.get("duration", 1))
    return jsonify(db_check(studio, date, duration))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
