import json
from database import check_availability, create_booking, get_booking, cancel_booking


# TOOL DEFINITIONS

# This is the dictionary we send to the LLM so it knows about available tools.
# The LLM reads the description and decides when to call each tool.
# Each tool has:
#   - name:        function name
#   - description: explanation for the LLM about when to use it
#   - parameters:  required data (JSON Schema)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check available time slots for a studio on a specific date. "
                "Call this when the client asks about availability, free times, "
                "or wants to know when a studio is open. "
                "Also call this before creating a booking to show the client their options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "studio": {
                        "type": "string",
                        "description": "Studio letter: A, B, C (New Cairo) or D, E, F (Dokki). Ask the client if not mentioned.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Convert relative dates like 'tomorrow' or 'Monday' to this format.",
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
            "name": "create_booking",
            "description": (
                "Create a confirmed booking for a studio. "
                "Only call this after the client has confirmed: their name, phone number, "
                "studio, date, start time, and duration. "
                "Do NOT call this unless all required info is collected and client confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Full name of the client.",
                    },
                    "phone": {
                        "type": "string",
                        "description": "Client phone number.",
                    },
                    "studio": {
                        "type": "string",
                        "description": "Studio letter: A, B, C, D, E, or F.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Start hour in 24h format (9 to 18). e.g. 14 for 2 PM.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Number of hours to book.",
                    },
                },
                "required": ["client_name", "phone", "studio", "date", "hour", "duration"],
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
                        "description": "The booking ID given to the client when they booked.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": (
                "Cancel an existing confirmed booking. "
                "Call this only after the client explicitly confirms they want to cancel. "
                "Always ask for confirmation before cancelling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID to cancel.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
]


# TOOL EXECUTOR

# When the LLM decides to call a tool, it sends us:
#   - tool_name:  function name
#   - tool_args:  the data (JSON)
# execute_tool runs the actual function and returns the result as JSON string


def execute_tool(tool_name: str, tool_args: dict, session_id: str) -> str:
    """
    Executes the tool requested by the LLM and returns the result as JSON string.
    session_id is needed only in create_booking to link the booking to the chat.
    """
    try:
        if tool_name == "check_availability":
            result = check_availability(
                studio=tool_args["studio"],
                target_date=tool_args["date"],
                duration=tool_args.get("duration", 1),
            )

        elif tool_name == "create_booking":
            result = create_booking(
                session_id=session_id,
                client_name=tool_args["client_name"],
                phone=tool_args["phone"],
                studio=tool_args["studio"],
                target_date=tool_args["date"],
                hour=tool_args["hour"],
                duration=tool_args["duration"],
            )

        elif tool_name == "get_booking":
            result = get_booking(booking_id=tool_args["booking_id"])

        elif tool_name == "cancel_booking":
            result = cancel_booking(booking_id=tool_args["booking_id"])

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        # If an error occurs during execution, return it to the LLM to explain to the client
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False)
