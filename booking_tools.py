import json
import re
from database import check_availability, get_booking


# ─────────────────────────────────────────────────────────────────────────────
#  TOOL DEFINITIONS
#  Only two tools remain:
#    - check_availability: show free slots for a studio on a date
#    - get_booking:        look up an existing booking by ID
#
#  create_booking and cancel_booking are intentionally removed.
#  If a client wants to book, the LLM will redirect them to the app or phone.
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check available time slots for a studio on a specific date. "
                "Call this when the client asks about availability, free times, "
                "or wants to know when a studio is open. "
                "Studios A, B, C are in New Cairo. Studios D, E, F are in Dokki."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "studio": {
                        "type": "string",
                        "description": (
                            "Studio letter: A, B, C (New Cairo) or D, E, F (Dokki). "
                            "A/D = Photography, B/E = Video, C/F = Podcast. "
                            "Ask the client if not mentioned."
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Date in YYYY-MM-DD format. "
                            "Convert relative dates like 'tomorrow' or 'Monday' to this format. "
                            "Today's date is in the system prompt."
                        ),
                    },
                    "duration": {
                        "type": "integer",
                        "description": "How many hours the client needs. Default is 1 if not mentioned.",
                        "default": 1,
                    },
                },
                "required": ["studio", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": (
                "Retrieve details of an existing booking using its ID. "
                "Call this when the client mentions a booking ID or asks about their booking status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID the client received when they booked.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict, session_id: str) -> str:
    """
    Executes the tool requested by the LLM and returns the result as a JSON string.
    Only check_availability and get_booking are supported.
    """
    try:
        if tool_name == "check_availability":
            # Normalize studio — extract single letter A-F
            # Handles: "A", "Studio A", "استوديو B", "photography" etc.
            studio_raw = tool_args.get("studio", "").strip()
            match = re.search(r"[A-Fa-f]", studio_raw)
            studio = match.group(0).upper() if match else studio_raw.upper()

            result = check_availability(
                studio=studio,
                target_date=tool_args["date"],
                duration=tool_args.get("duration", 1),
            )
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "get_booking":
            result = get_booking(booking_id=tool_args["booking_id"])
            return json.dumps(result, ensure_ascii=False)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
