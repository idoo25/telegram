#!/usr/bin/env python3
"""Quick script to check database contents."""
import sqlite3
import os

DB_PATH = 'telegram.db'

if not os.path.exists(DB_PATH):
    print(f"Database not found: {DB_PATH}")
    exit(1)

try:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get total count
    total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Total messages in database: {total}")

    # Get date range
    date_range = conn.execute("""
        SELECT MIN(date) as earliest, MAX(date) as latest
        FROM messages
    """).fetchone()
    print(f"Date range: {date_range['earliest']} to {date_range['latest']}")
    print()

    # Show 50 newest messages
    print("=" * 60)
    print("50 NEWEST MESSAGES:")
    print("=" * 60)

    rows = conn.execute("""
        SELECT date, from_name, text_plain
        FROM messages
        ORDER BY date DESC
        LIMIT 50
    """).fetchall()

    for row in rows:
        text = (row['text_plain'] or '')[:80].replace('\n', ' ')
        name = row['from_name'] or 'Unknown'
        print(f"{row['date']} | {name}: {text}")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
