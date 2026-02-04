#!/usr/bin/env python3
"""
Telegram JSON Chat Indexer
Parses Telegram export JSON and indexes into SQLite with FTS5 for full-text search.

Usage:
    python indexer.py <json_file> [--db <database_file>]
    python indexer.py result.json --db telegram.db
"""

import json
import sqlite3
import argparse
import os
from pathlib import Path
from typing import Any, Generator


def flatten_text(text_field: Any) -> str:
    """
    Flatten the text field which can be either a string or array of mixed content.

    Examples:
        "hello" -> "hello"
        ["hello", {"type": "link", "text": "url"}, " world"] -> "hello url world"
    """
    if isinstance(text_field, str):
        return text_field

    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
        return ''.join(parts)

    return ''


def extract_entities(text_entities: list) -> list[dict]:
    """Extract typed entities (links, mentions, etc.) from text_entities array."""
    entities = []
    for entity in text_entities or []:
        if isinstance(entity, dict):
            entity_type = entity.get('type', 'plain')
            if entity_type != 'plain':  # Skip plain text
                entities.append({
                    'type': entity_type,
                    'value': entity.get('text', '')
                })
    return entities


def parse_message(msg: dict) -> dict | None:
    """Parse a single message from Telegram JSON format."""
    if msg.get('type') != 'message':
        return None

    text_plain = flatten_text(msg.get('text', ''))
    entities = extract_entities(msg.get('text_entities', []))

    has_links = any(e['type'] == 'link' for e in entities)
    has_mentions = any(e['type'] == 'mention' for e in entities)

    return {
        'id': msg.get('id'),
        'type': msg.get('type', 'message'),
        'date': msg.get('date'),
        'date_unixtime': int(msg.get('date_unixtime', 0)) if msg.get('date_unixtime') else None,
        'from_name': msg.get('from', ''),
        'from_id': msg.get('from_id', ''),
        'reply_to_message_id': msg.get('reply_to_message_id'),
        'forwarded_from': msg.get('forwarded_from'),
        'forwarded_from_id': msg.get('forwarded_from_id'),
        'text_plain': text_plain,
        'has_media': 1 if msg.get('photo') or msg.get('file') or msg.get('media_type') else 0,
        'has_photo': 1 if msg.get('photo') else 0,
        'has_links': 1 if has_links else 0,
        'has_mentions': 1 if has_mentions else 0,
        'is_edited': 1 if msg.get('edited') else 0,
        'edited_unixtime': int(msg.get('edited_unixtime', 0)) if msg.get('edited_unixtime') else None,
        'photo_file_size': msg.get('photo_file_size'),
        'photo_width': msg.get('width'),
        'photo_height': msg.get('height'),
        'raw_json': json.dumps(msg, ensure_ascii=False),
        'entities': entities
    }


def load_json_messages(json_path: str) -> Generator[dict, None, None]:
    """Load messages from Telegram export JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats: direct array or {"messages": [...]}
    messages = data if isinstance(data, list) else data.get('messages', [])

    for msg in messages:
        parsed = parse_message(msg)
        if parsed:
            yield parsed


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Read and execute schema
    schema_path = Path(__file__).parent / 'schema.sql'
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
    else:
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    return conn


def index_messages(conn: sqlite3.Connection, messages: Generator[dict, None, None]) -> dict:
    """Index messages into the database."""
    cursor = conn.cursor()
    stats = {
        'messages': 0,
        'entities': 0,
        'users': {},
        'skipped': 0
    }

    for msg in messages:
        try:
            # Insert message
            cursor.execute('''
                INSERT OR REPLACE INTO messages (
                    id, type, date, date_unixtime, from_name, from_id,
                    reply_to_message_id, forwarded_from, forwarded_from_id,
                    text_plain, has_media, has_photo, has_links, has_mentions,
                    is_edited, edited_unixtime, photo_file_size, photo_width,
                    photo_height, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                msg['id'], msg['type'], msg['date'], msg['date_unixtime'],
                msg['from_name'], msg['from_id'], msg['reply_to_message_id'],
                msg['forwarded_from'], msg['forwarded_from_id'], msg['text_plain'],
                msg['has_media'], msg['has_photo'], msg['has_links'],
                msg['has_mentions'], msg['is_edited'], msg['edited_unixtime'],
                msg['photo_file_size'], msg['photo_width'], msg['photo_height'],
                msg['raw_json']
            ))
            stats['messages'] += 1

            # Insert entities
            for entity in msg['entities']:
                cursor.execute('''
                    INSERT INTO entities (message_id, type, value)
                    VALUES (?, ?, ?)
                ''', (msg['id'], entity['type'], entity['value']))
                stats['entities'] += 1

            # Track users
            user_id = msg['from_id']
            if user_id:
                if user_id not in stats['users']:
                    stats['users'][user_id] = {
                        'display_name': msg['from_name'],
                        'first_seen': msg['date_unixtime'],
                        'last_seen': msg['date_unixtime'],
                        'count': 0
                    }
                stats['users'][user_id]['count'] += 1
                if msg['date_unixtime']:
                    if msg['date_unixtime'] < stats['users'][user_id]['first_seen']:
                        stats['users'][user_id]['first_seen'] = msg['date_unixtime']
                    if msg['date_unixtime'] > stats['users'][user_id]['last_seen']:
                        stats['users'][user_id]['last_seen'] = msg['date_unixtime']

        except Exception as e:
            print(f"Error indexing message {msg.get('id')}: {e}")
            stats['skipped'] += 1

    # Update users table
    for user_id, user_data in stats['users'].items():
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, display_name, first_seen, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, user_data['display_name'], user_data['first_seen'],
              user_data['last_seen'], user_data['count']))

    conn.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description='Index Telegram JSON export to SQLite')
    parser.add_argument('json_file', help='Path to Telegram export JSON file')
    parser.add_argument('--db', default='telegram.db', help='SQLite database path (default: telegram.db)')
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1

    print(f"Initializing database: {args.db}")
    conn = init_database(args.db)

    print(f"Loading and indexing: {args.json_file}")
    messages = load_json_messages(args.json_file)
    stats = index_messages(conn, messages)

    print(f"\nIndexing complete!")
    print(f"  Messages indexed: {stats['messages']}")
    print(f"  Entities extracted: {stats['entities']}")
    print(f"  Unique users: {len(stats['users'])}")
    print(f"  Skipped: {stats['skipped']}")

    conn.close()
    print(f"\nDatabase saved to: {args.db}")
    return 0


if __name__ == '__main__':
    exit(main())
