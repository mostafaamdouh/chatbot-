import sqlite3
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "genz.db"

# Working hours: 9 AM to 7 PM
# Last booking slot starts at 6 PM (18:00) so it ends by 7 PM (19:00)
WORK_START = 9
WORK_END   = 19  # exclusive upper bound for slot generation
LAST_START = 18  # last valid start hour for a booking

# Available studios and their hourly rates
STUDIOS = {
    "A": {"branch": "New Cairo", "type": "photography",  "rate": 1500},
    "B": {"branch": "New Cairo", "type": "video",        "rate": 2500},
    "C": {"branch": "New Cairo", "type": "podcast",      "rate": 800},
    "D": {"branch": "Dokki",     "type": "photography",  "rate": 1500},
    "E": {"branch": "Dokki",     "type": "video",        "rate": 2500},
    "F": {"branch": "Dokki",     "type": "podcast",      "rate": 800},
}


def get_connection():
    """
    Open a new connection per request for thread safety.
    row_factory = sqlite3.Row lets us access columns by name (row["hour"]).
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bookings (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                client_name TEXT NOT NULL,
                phone       TEXT NOT NULL,
                studio      TEXT NOT NULL,
                date        TEXT NOT NULL,
                hour        INTEGER NOT NULL,
                duration    INTEGER NOT NULL,
                total_price INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'confirmed',
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS studio_schedule (
                studio       TEXT NOT NULL,
                date         TEXT NOT NULL,
                hour         INTEGER NOT NULL,
                is_available INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (studio, date, hour)
            );
        """)
        conn.commit()
        print("[DB] Database initialized successfully.")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def ensure_schedule_exists(studio: str, target_date: str):
    """
    Lazily generate hourly slots for a studio on a date if they don't exist.
    We don't pre-fill the entire year — only generate when actually needed.
    """
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM studio_schedule WHERE studio=? AND date=?",
            (studio, target_date)
        ).fetchone()

        if existing["cnt"] == 0:
            slots = [(studio, target_date, h, 1) for h in range(WORK_START, WORK_END)]
            conn.executemany(
                "INSERT OR IGNORE INTO studio_schedule (studio, date, hour, is_available) VALUES (?,?,?,?)",
                slots
            )
            conn.commit()
    finally:
        conn.close()


def _mark_slots(conn, studio: str, target_date: str, hour: int, duration: int, available: int):
    """Mark one or more consecutive hour slots as booked (0) or free (1)."""
    for h in range(hour, hour + duration):
        conn.execute(
            "UPDATE studio_schedule SET is_available=? WHERE studio=? AND date=? AND hour=?",
            (available, studio, target_date, h)
        )


# ─────────────────────────────────────────────────────────────────────────────
#  AVAILABILITY
# ─────────────────────────────────────────────────────────────────────────────

def check_availability(studio: str, target_date: str, duration: int = 1) -> dict:
    """
    Return all valid start slots for a studio+date+duration combination.

    A slot is valid only if:
    - It is marked as available in studio_schedule
    - All 'duration' consecutive hours starting from it are also available
    - The booking would end by WORK_END (19:00)
    """
    studio = studio.upper()
    if studio not in STUDIOS:
        return {"error": f"Studio '{studio}' not found. Valid studios: A, B, C, D, E, F"}

    ensure_schedule_exists(studio, target_date)

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT hour FROM studio_schedule WHERE studio=? AND date=? AND is_available=1 ORDER BY hour",
            (studio, target_date)
        ).fetchall()

        available_hours = {r["hour"] for r in rows}

        valid_slots = []
        for h in sorted(available_hours):
            # Slot must end by closing time
            if h + duration > WORK_END:
                continue
            # All hours in the block must be free
            needed = set(range(h, h + duration))
            if needed.issubset(available_hours):
                valid_slots.append(f"{h:02d}:00")

        return {
            "available":       len(valid_slots) > 0,
            "slots":           valid_slots,
            "studio":          studio,
            "branch":          STUDIOS[studio]["branch"],
            "type":            STUDIOS[studio]["type"],
            "date":            target_date,
            "hourly_rate":     STUDIOS[studio]["rate"],
            "estimated_price": STUDIOS[studio]["rate"] * duration,
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  BOOKING
# ─────────────────────────────────────────────────────────────────────────────

def create_booking(
    session_id: str,
    client_name: str,
    phone: str,
    studio: str,
    target_date: str,
    hour: int,
    duration: int,
) -> dict:
    """
    Create a confirmed booking after validating all inputs.

    Validations performed here (last line of defense after booking_tools resolvers):
    - Studio letter must be valid
    - hour must be within working hours
    - booking must end by closing time
    - no conflicting slots (race condition check)
    """
    studio = studio.upper()
    if studio not in STUDIOS:
        return {"success": False, "error": f"Studio '{studio}' not found."}

    # ── Hour range validation ────────────────────────────────────────────────
    # booking_tools.resolve_hour() normally handles this, but we double-check
    # here as a safety net in case the function is called directly.
    if not isinstance(hour, int) or hour < WORK_START or hour > LAST_START:
        return {
            "success": False,
            "error": (
                f"Invalid start hour: {hour}. "
                f"Bookings must start between 09:00 (9 AM) and 18:00 (6 PM)."
            ),
        }

    # ── Duration must fit within working hours ───────────────────────────────
    if hour + duration > WORK_END:
        max_duration = WORK_END - hour
        return {
            "success": False,
            "error": (
                f"A {duration}-hour booking starting at {hour:02d}:00 would end after closing time. "
                f"Maximum duration from {hour:02d}:00 is {max_duration} hour(s)."
            ),
        }

    ensure_schedule_exists(studio, target_date)

    conn = get_connection()
    try:
        # ── Race condition check ─────────────────────────────────────────────
        needed_hours = list(range(hour, hour + duration))
        placeholders = ",".join("?" * len(needed_hours))
        booked = conn.execute(
            f"SELECT COUNT(*) as cnt FROM studio_schedule "
            f"WHERE studio=? AND date=? AND hour IN ({placeholders}) AND is_available=0",
            [studio, target_date] + needed_hours
        ).fetchone()

        if booked["cnt"] > 0:
            return {
                "success": False,
                "error": "That slot was just booked by someone else. Please choose another time.",
            }

        # ── Create the booking ───────────────────────────────────────────────
        booking_id  = str(uuid.uuid4())[:8].upper()
        total_price = STUDIOS[studio]["rate"] * duration
        created_at  = datetime.now().isoformat()

        conn.execute(
            """
            INSERT INTO bookings
                (id, session_id, client_name, phone, studio, date, hour, duration, total_price, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (booking_id, session_id, client_name, phone, studio,
             target_date, hour, duration, total_price, "confirmed", created_at)
        )
        _mark_slots(conn, studio, target_date, hour, duration, available=0)
        conn.commit()

        return {
            "success":     True,
            "booking_id":  booking_id,
            "client_name": client_name,
            "phone":       phone,
            "studio":      studio,
            "branch":      STUDIOS[studio]["branch"],
            "type":        STUDIOS[studio]["type"],
            "date":        target_date,
            "hour":        f"{hour:02d}:00",
            "duration":    duration,
            "total_price": total_price,
            "status":      "confirmed",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  GET BOOKING
# ─────────────────────────────────────────────────────────────────────────────

def get_booking(booking_id: str) -> dict:
    """Return full booking details by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM bookings WHERE id=?",
            (booking_id.upper(),)
        ).fetchone()

        if not row:
            return {"found": False, "error": f"No booking found with ID '{booking_id}'."}

        return {
            "found":       True,
            "booking_id":  row["id"],
            "client_name": row["client_name"],
            "phone":       row["phone"],
            "studio":      row["studio"],
            "branch":      STUDIOS.get(row["studio"], {}).get("branch", ""),
            "type":        STUDIOS.get(row["studio"], {}).get("type", ""),
            "date":        row["date"],
            "hour":        f"{row['hour']:02d}:00",
            "duration":    row["duration"],
            "total_price": row["total_price"],
            "status":      row["status"],
            "created_at":  row["created_at"],
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  CANCEL BOOKING
# ─────────────────────────────────────────────────────────────────────────────

def cancel_booking(booking_id: str) -> dict:
    """
    Cancel a booking and restore its slots to available.
    Only confirmed bookings can be cancelled.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM bookings WHERE id=? AND status='confirmed'",
            (booking_id.upper(),)
        ).fetchone()

        if not row:
            return {"success": False, "error": f"No active booking found with ID '{booking_id}'."}

        conn.execute(
            "UPDATE bookings SET status='cancelled' WHERE id=?",
            (booking_id.upper(),)
        )
        _mark_slots(conn, row["studio"], row["date"], row["hour"], row["duration"], available=1)
        conn.commit()

        return {
            "success":    True,
            "booking_id": booking_id.upper(),
            "message": (
                f"Booking cancelled. Studio {row['studio']} on {row['date']} "
                f"at {row['hour']:02d}:00 is now available again."
            ),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_bookings(status: str = None) -> list:
    """Return all bookings, optionally filtered by status."""
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM bookings WHERE status=? ORDER BY date, hour", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bookings ORDER BY date, hour"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  INIT — runs automatically on import
# ─────────────────────────────────────────────────────────────────────────────
init_db()
