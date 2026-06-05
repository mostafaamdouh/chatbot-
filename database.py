import sqlite3
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "genz.db"

# working hours: 9 AM to 7 PM (last slot starts at 6 PM because booking is by the hour)

WORK_START = 9   # 9 AM
WORK_END   = 19  # 7 PM (last booking can start at 6 PM)

# available studios and their rates
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
    we open a new connection for each request to ensure thread safety.
    SQLite connections are not thread-safe by default, so we set check_same_thread=False.
    to prevent issues with multiple threads accessing the same connection, we create a new one for each request and close it afterward.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    # default row factory returns rows as tuples, but we change it to sqlite3.Row
    # so we can access columns by name (for example row["hour"] instead of row[5]).
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    create the necessary tables if they don't exist.
    we use executescript to run multiple SQL statements at once, which is more efficient than multiple execute calls.
    """
    conn = get_connection()
    try:
        conn.executescript("""
            -- ─── bookings table ───────────────────────────────────────────────
            -- Each row is one complete booking.
            CREATE TABLE IF NOT EXISTS bookings (
                id          TEXT PRIMARY KEY,   -- Unique UUID for each booking
                session_id  TEXT NOT NULL,      -- Chat session ID that created the booking
                client_name TEXT NOT NULL,      -- Client name
                phone       TEXT NOT NULL,      -- Phone number
                studio      TEXT NOT NULL,      -- A/B/C/D/E/F
                date        TEXT NOT NULL,      -- YYYY-MM-DD
                hour        INTEGER NOT NULL,   -- 9..18 (start hour)
                duration    INTEGER NOT NULL,   -- Number of hours (1..8)
                total_price INTEGER NOT NULL,   -- Total price in EGP
                status      TEXT NOT NULL DEFAULT 'confirmed', -- confirmed/cancelled
                created_at  TEXT NOT NULL       -- Booking creation timestamp
            );

            -- ─── schedule table (available slots) ─────────────────────────────
            -- Each row is one hour slot for one studio on one date.
            -- This is filled lazily when someone asks about a specific date.
            CREATE TABLE IF NOT EXISTS studio_schedule (
                studio       TEXT NOT NULL,   -- A/B/C/D/E/F
                date         TEXT NOT NULL,   -- YYYY-MM-DD
                hour         INTEGER NOT NULL, -- 9..18
                is_available INTEGER NOT NULL DEFAULT 1, -- 1=available 0=booked
                PRIMARY KEY (studio, date, hour)  -- no duplicate records for the same slot
            );
        """)
        conn.commit()
        print("[DB] Database initialized successfully.")
    finally:
        conn.close()


#  SCHEDULE GENERATION 

def ensure_schedule_exists(studio: str, target_date: str):
    """
    Ensure a schedule exists for the studio on the requested date.
    If it does not exist yet, create slots for the working hours (9 to 18).

    Why lazy generation?
    Instead of prepopulating an entire year of slots up front
    (which could be thousands of rows), we only generate slots when
    someone asks for that specific date.
    """
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM studio_schedule WHERE studio=? AND date=?",
            (studio, target_date)
        ).fetchone()

        if existing["cnt"] == 0:
            # create (18) slots for the day (from 9 to 18)
            slots = [
                (studio, target_date, hour, 1)
                for hour in range(WORK_START, WORK_END)
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO studio_schedule (studio, date, hour, is_available) VALUES (?,?,?,?)",
                slots
            )
            conn.commit()
    finally:
        conn.close()


def _mark_slots(conn, studio: str, target_date: str, hour: int, duration: int, available: int):
    """
    Helper that updates slot availability in the schedule table.
    available=0 → booked | available=1 → available again (for cancellations)
    """
    for h in range(hour, hour + duration):
        conn.execute(
            "UPDATE studio_schedule SET is_available=? WHERE studio=? AND date=? AND hour=?",
            (available, studio, target_date, h)
        )


#  AVAILABILITY 

def check_availability(studio: str, target_date: str, duration: int = 1) -> dict:
    """
    Return available time slots for a studio on a given date.

    The duration parameter removes slots that cannot fit the requested length.
    For example, if the customer needs 3 hours, the 17:00 slot is not valid
    because it would end after the closing hour.

    The return value contains availability, slot times, studio details, and pricing.
    """
    studio = studio.upper()
    if studio not in STUDIOS:
        return {"error": f"Studio {studio} not found. Available studios: A, B, C, D, E, F"}

    ensure_schedule_exists(studio, target_date)

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT hour FROM studio_schedule
            WHERE studio=? AND date=? AND is_available=1
            ORDER BY hour
            """,
            (studio, target_date)
        ).fetchall()

        available_hours = [r["hour"] for r in rows]

        # Filter out slots that cannot fit the requested duration
        valid_slots = []
        for h in available_hours:
            if h + duration <= WORK_END:
                # Confirm that all required duration hours are available
                needed = set(range(h, h + duration))
                if needed.issubset(set(available_hours)):
                    valid_slots.append(f"{h:02d}:00")

        return {
            "available": len(valid_slots) > 0,
            "slots": valid_slots,
            "studio": studio,
            "branch": STUDIOS[studio]["branch"],
            "type": STUDIOS[studio]["type"],
            "date": target_date,
            "hourly_rate": STUDIOS[studio]["rate"],
            "estimated_price": STUDIOS[studio]["rate"] * duration,
        }
    finally:
        conn.close()


#  BOOKING 

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
    Create a new booking if the requested time is still available.

    Performs a double-check of availability inside the transaction to prevent
    a race condition: if two requests arrive for the same slot at the same time,
    the first one should succeed and the second one should return a clear error.

    The return value includes booking confirmation details.
    """
    studio = studio.upper()
    if studio not in STUDIOS:
        return {"success": False, "error": f"Studio {studio} not found."}

    ensure_schedule_exists(studio, target_date)

    conn = get_connection()
    try:
        # double-check: make sure all requested slots are still available
        needed_hours = list(range(hour, hour + duration))
        placeholders = ",".join("?" * len(needed_hours))
        booked = conn.execute(
            f"""
            SELECT COUNT(*) as cnt FROM studio_schedule
            WHERE studio=? AND date=? AND hour IN ({placeholders}) AND is_available=0
            """,
            [studio, target_date] + needed_hours
        ).fetchone()

        if booked["cnt"] > 0:
            return {
                "success": False,
                "error": "Sorry, that slot was just booked. Please choose another time.",
            }

        # All checks passed — create the booking
        booking_id  = str(uuid.uuid4())[:8].upper()   # Short, easy-to-remember ID
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

        # Mark the slots as booked in the schedule table
        _mark_slots(conn, studio, target_date, hour, duration, available=0)
        conn.commit()

        return {
            "success": True,
            "booking_id": booking_id,
            "client_name": client_name,
            "phone": phone,
            "studio": studio,
            "branch": STUDIOS[studio]["branch"],
            "type": STUDIOS[studio]["type"],
            "date": target_date,
            "hour": f"{hour:02d}:00",
            "duration": duration,
            "total_price": total_price,
            "status": "confirmed",
        }
    finally:
        conn.close()


#  GET BOOKING 

def get_booking(booking_id: str) -> dict:
    """
    Return booking details by booking ID.
    Useful when a customer asks, "What are my booking details?"
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM bookings WHERE id=?",
            (booking_id.upper(),)
        ).fetchone()

        if not row:
            return {"found": False, "error": f"No booking found for ID {booking_id}"}

        return {
            "found": True,
            "booking_id": row["id"],
            "client_name": row["client_name"],
            "phone": row["phone"],
            "studio": row["studio"],
            "branch": STUDIOS.get(row["studio"], {}).get("branch", ""),
            "date": row["date"],
            "hour": f"{row['hour']:02d}:00",
            "duration": row["duration"],
            "total_price": row["total_price"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


#  CANCEL BOOKING 

def cancel_booking(booking_id: str) -> dict:
    """
    Cancel a booking and free the corresponding slots so someone else can book them.

    Important: we also update studio_schedule so check_availability remains correct.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM bookings WHERE id=? AND status='confirmed'",
            (booking_id.upper(),)
        ).fetchone()

        if not row:
            return {"success": False, "error": f"No active booking found with ID {booking_id}"}

        # Update the booking status to cancelled
        conn.execute(
            "UPDATE bookings SET status='cancelled' WHERE id=?",
            (booking_id.upper(),)
        )

        # Restore the slots in the schedule table (is_available=1)
        _mark_slots(conn, row["studio"], row["date"], row["hour"], row["duration"], available=1)
        conn.commit()

        return {
            "success": True,
            "booking_id": booking_id.upper(),
            "message": f"Booking cancelled successfully. The slot on {row['date']} at {row['hour']:02d}:00 is now available again.",
        }
    finally:
        conn.close()


#  ADMIN HELPERS 

def get_all_bookings(status: str = None) -> list:
    """Return all bookings — useful for the admin dashboard."""
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM bookings WHERE status=? ORDER BY date, hour",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bookings ORDER BY date, hour"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


#  INIT DATABASE
# This runs automatically when any other module imports database.py
init_db()
