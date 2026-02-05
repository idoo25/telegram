#!/usr/bin/env python3
"""
Daily Telegram Sync Script
===========================
Automatically syncs new messages from Telegram group to the analytics system.

What it does:
1. Connects to Telegram via Telethon
2. Fetches messages from the last 36 hours (12h overlap for safety)
3. Adds new messages to telegram.db (duplicates ignored)
4. Generates embeddings for new messages locally
5. Adds embeddings to embeddings.db (duplicates ignored)

Usage:
    First time:  python daily_sync.py --setup
    Daily run:   python daily_sync.py
    Custom hours: python daily_sync.py --hours 48

Automation:
    Windows Task Scheduler: python daily_sync.py
    Linux cron: 0 3 * * * cd /path/to/telegram && python daily_sync.py >> sync.log 2>&1
"""

import os
import sys
import json
import time
import sqlite3
import asyncio
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Setup logging
LOG_FILE = Path(__file__).parent / 'sync.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
log = logging.getLogger('daily_sync')

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'telegram.db'
EMBEDDINGS_DB_PATH = BASE_DIR / 'embeddings.db'
SESSION_FILE = BASE_DIR / 'telegram_session'
CONFIG_FILE = BASE_DIR / 'sync_config.json'

# ==========================================
# CONFIGURATION
# ==========================================

def load_config() -> dict:
    """Load configuration from sync_config.json."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save configuration to sync_config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def setup_config():
    """Interactive setup for first time configuration."""
    print("=" * 50)
    print("  Telegram Daily Sync - Setup")
    print("=" * 50)
    print()
    print("You need Telegram API credentials.")
    print("Get them from: https://my.telegram.org/apps")
    print()

    api_id = input("API ID: ").strip()
    api_hash = input("API Hash: ").strip()
    group = input("Group username or ID (e.g., @mygroup or -1001234567890): ").strip()

    config = {
        'api_id': int(api_id),
        'api_hash': api_hash,
        'group': group,
        'hours': 36
    }

    save_config(config)
    print(f"\nConfiguration saved to {CONFIG_FILE}")
    print("Now run: python daily_sync.py")
    print("(First run will ask you to log in to Telegram)")
    return config


# ==========================================
# TELETHON: FETCH MESSAGES
# ==========================================

def telethon_message_to_json(message) -> dict | None:
    """
    Convert a Telethon message object to Telegram Desktop export JSON format.
    This ensures compatibility with the existing parse_message() in indexer.py.
    """
    from telethon.tl.types import (
        MessageEntityUrl, MessageEntityTextUrl,
        MessageEntityMention, MessageEntityMentionName,
        MessageEntityBold, MessageEntityItalic,
        MessageEntityCode, MessageEntityPre,
        MessageEntityHashtag, MessageEntityEmail,
        MessageEntityPhone, MessageEntityBotCommand,
    )

    if message.text is None and message.raw_text is None:
        return None

    text = message.raw_text or ''
    if not text.strip():
        # Skip empty messages (media-only, service messages, etc.)
        return None

    # Build text_entities in Telegram Desktop export format
    text_entities = []
    if message.entities:
        for entity in message.entities:
            start = entity.offset
            end = entity.offset + entity.length
            entity_text = text[start:end]

            entity_type = 'plain'
            if isinstance(entity, (MessageEntityUrl,)):
                entity_type = 'link'
            elif isinstance(entity, MessageEntityTextUrl):
                entity_type = 'text_link'
                entity_text = entity.url
            elif isinstance(entity, (MessageEntityMention, MessageEntityMentionName)):
                entity_type = 'mention'
            elif isinstance(entity, MessageEntityBold):
                entity_type = 'bold'
            elif isinstance(entity, MessageEntityItalic):
                entity_type = 'italic'
            elif isinstance(entity, (MessageEntityCode, MessageEntityPre)):
                entity_type = 'code'
            elif isinstance(entity, MessageEntityHashtag):
                entity_type = 'hashtag'
            elif isinstance(entity, MessageEntityEmail):
                entity_type = 'email'
            elif isinstance(entity, MessageEntityPhone):
                entity_type = 'phone'
            elif isinstance(entity, MessageEntityBotCommand):
                entity_type = 'bot_command'

            if entity_type != 'plain':
                text_entities.append({
                    'type': entity_type,
                    'text': entity_text
                })

    # Get sender info
    sender = message.sender
    from_name = ''
    from_id = ''

    if sender:
        if hasattr(sender, 'first_name'):
            # User
            parts = [sender.first_name or '']
            if sender.last_name:
                parts.append(sender.last_name)
            from_name = ' '.join(parts).strip()
            from_id = f'user{sender.id}'
        elif hasattr(sender, 'title'):
            # Channel/Group
            from_name = sender.title or ''
            from_id = f'channel{sender.id}'

    # Handle forwarded messages
    forwarded_from = None
    forwarded_from_id = None
    if message.forward:
        fwd = message.forward
        if fwd.sender:
            if hasattr(fwd.sender, 'first_name'):
                parts = [fwd.sender.first_name or '']
                if fwd.sender.last_name:
                    parts.append(fwd.sender.last_name)
                forwarded_from = ' '.join(parts).strip()
                forwarded_from_id = f'user{fwd.sender.id}'
            elif hasattr(fwd.sender, 'title'):
                forwarded_from = fwd.sender.title
                forwarded_from_id = f'channel{fwd.sender.id}'
        elif fwd.sender_name:
            forwarded_from = fwd.sender_name

    # Photo info
    photo_info = {}
    has_photo = False
    has_media = False

    if message.photo:
        has_photo = True
        has_media = True
        if hasattr(message.photo, 'sizes') and message.photo.sizes:
            largest = message.photo.sizes[-1]
            if hasattr(largest, 'w'):
                photo_info['width'] = largest.w
                photo_info['height'] = largest.h
    elif message.document or message.video or message.audio:
        has_media = True

    # Reply info
    reply_to = None
    if message.reply_to:
        reply_to = message.reply_to.reply_to_msg_id

    # Date handling
    msg_date = message.date.replace(tzinfo=None) if message.date else datetime.utcnow()

    # Build the JSON in Telegram Desktop export format
    msg_json = {
        'id': message.id,
        'type': 'message',
        'date': msg_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'date_unixtime': str(int(msg_date.replace(tzinfo=timezone.utc).timestamp())),
        'from': from_name,
        'from_id': from_id,
        'text': text,
        'text_entities': text_entities,
    }

    if reply_to:
        msg_json['reply_to_message_id'] = reply_to
    if forwarded_from:
        msg_json['forwarded_from'] = forwarded_from
    if forwarded_from_id:
        msg_json['forwarded_from_id'] = forwarded_from_id
    if has_photo:
        msg_json['photo'] = '(photo)'
        msg_json.update(photo_info)
    if has_media and not has_photo:
        msg_json['media_type'] = 'document'
    if message.edit_date:
        edit_date = message.edit_date.replace(tzinfo=None)
        msg_json['edited'] = edit_date.strftime('%Y-%m-%dT%H:%M:%S')
        msg_json['edited_unixtime'] = str(int(edit_date.replace(tzinfo=timezone.utc).timestamp()))

    return msg_json


async def fetch_messages(config: dict, hours: int = 36) -> list[dict]:
    """
    Fetch messages from the last N hours using Telethon.
    Returns messages in Telegram Desktop export JSON format.
    """
    from telethon import TelegramClient

    api_id = config['api_id']
    api_hash = config['api_hash']
    group = config['group']

    client = TelegramClient(str(SESSION_FILE), api_id, api_hash)
    await client.start()

    log.info(f"Connected to Telegram")

    # Resolve group
    entity = await client.get_entity(group)
    log.info(f"Fetching from: {getattr(entity, 'title', group)}")

    # Calculate time window
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    log.info(f"Fetching messages from last {hours} hours (since {since.strftime('%Y-%m-%d %H:%M')} UTC)")

    # Fetch messages
    messages_json = []
    count = 0

    async for message in client.iter_messages(entity, offset_date=None, reverse=False):
        # Stop when we've gone past our time window
        if message.date < since:
            break

        count += 1
        msg_json = telethon_message_to_json(message)
        if msg_json:
            messages_json.append(msg_json)

    await client.disconnect()

    log.info(f"Fetched {count} messages, {len(messages_json)} with text content")
    return messages_json


# ==========================================
# DATABASE: INDEX NEW MESSAGES
# ==========================================

def index_messages(messages_json: list[dict]) -> dict:
    """
    Add new messages to telegram.db using IncrementalIndexer.
    Duplicates are automatically ignored.
    """
    if not messages_json:
        return {'new_messages': 0, 'duplicates': 0}

    from indexer import IncrementalIndexer

    log.info(f"Indexing {len(messages_json)} messages into telegram.db...")

    indexer = IncrementalIndexer(str(DB_PATH))
    stats = indexer.update_from_json_data({'messages': messages_json}, show_progress=True)
    indexer.close()

    log.info(f"Indexing done: {stats['new_messages']} new, {stats['duplicates']} duplicates")
    return stats


# ==========================================
# EMBEDDINGS: GENERATE FOR NEW MESSAGES
# ==========================================

def generate_embeddings(messages_json: list[dict]) -> dict:
    """
    Generate embeddings for new messages and add to embeddings.db.
    Only processes messages that don't already have embeddings.
    """
    if not os.path.exists(EMBEDDINGS_DB_PATH):
        log.warning(f"embeddings.db not found at {EMBEDDINGS_DB_PATH}. Skipping embeddings.")
        return {'new_embeddings': 0, 'skipped': 0}

    import numpy as np

    # Find which messages don't have embeddings yet
    emb_conn = sqlite3.connect(str(EMBEDDINGS_DB_PATH))
    existing_ids = set()
    for row in emb_conn.execute("SELECT message_id FROM embeddings"):
        existing_ids.add(row[0])

    # Filter to messages that:
    # 1. Have text content
    # 2. Text is longer than 10 chars
    # 3. Don't already have embeddings
    new_messages = []
    for msg in messages_json:
        msg_id = msg.get('id')
        text = msg.get('text', '')
        if isinstance(text, list):
            # Handle complex text format
            text = ''.join(
                part if isinstance(part, str) else part.get('text', '')
                for part in text
            )

        if msg_id and msg_id not in existing_ids and text and len(text.strip()) > 10:
            new_messages.append({
                'id': msg_id,
                'from_name': msg.get('from', ''),
                'text': text
            })

    if not new_messages:
        emb_conn.close()
        log.info("No new messages need embeddings")
        return {'new_embeddings': 0, 'skipped': len(messages_json)}

    log.info(f"Generating embeddings for {len(new_messages)} new messages...")

    # Load model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Generate embeddings
    texts = [m['text'][:500] for m in new_messages]  # Max 500 chars per message
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True,
                              batch_size=64)

    # Insert into embeddings.db
    data = []
    for i, msg in enumerate(new_messages):
        emb_blob = embeddings[i].astype(np.float32).tobytes()
        data.append((
            msg['id'],
            msg['from_name'],
            msg['text'][:100],  # Preview
            emb_blob
        ))

    emb_conn.executemany(
        "INSERT OR IGNORE INTO embeddings (message_id, from_name, text_preview, embedding) VALUES (?, ?, ?, ?)",
        data
    )
    emb_conn.commit()

    # Verify
    total = emb_conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    emb_conn.close()

    log.info(f"Added {len(data)} new embeddings. Total embeddings: {total:,}")
    return {'new_embeddings': len(data), 'total_embeddings': total}


# ==========================================
# MAIN
# ==========================================

def run_sync(hours: int = 36, skip_embeddings: bool = False):
    """Run the full sync pipeline."""
    start_time = time.time()

    log.info("=" * 50)
    log.info("Starting daily sync")
    log.info("=" * 50)

    # Load config
    config = load_config()
    if not config:
        log.error("No configuration found. Run: python daily_sync.py --setup")
        sys.exit(1)

    # Step 1: Fetch messages from Telegram
    log.info("[1/3] Fetching messages from Telegram...")
    messages_json = asyncio.run(fetch_messages(config, hours=hours))

    if not messages_json:
        log.info("No messages found in the time window. Done.")
        return

    # Step 2: Index into telegram.db
    log.info("[2/3] Indexing new messages...")
    index_stats = index_messages(messages_json)

    # Step 3: Generate embeddings
    if skip_embeddings:
        log.info("[3/3] Skipping embeddings (--skip-embeddings)")
        emb_stats = {'new_embeddings': 0}
    else:
        log.info("[3/3] Generating embeddings for new messages...")
        emb_stats = generate_embeddings(messages_json)

    # Step 4: Notify running server to reload embeddings
    if emb_stats.get('new_embeddings', 0) > 0:
        try:
            import urllib.request
            req = urllib.request.urlopen('http://localhost:5000/api/embeddings/reload', timeout=5)
            log.info("Server notified to reload embeddings")
        except Exception:
            log.info("Server not running or unreachable - embeddings will load on next restart")

    # Summary
    elapsed = time.time() - start_time
    log.info("=" * 50)
    log.info("Sync complete!")
    log.info(f"  Messages fetched:    {len(messages_json)}")
    log.info(f"  New to DB:           {index_stats.get('new_messages', 0)}")
    log.info(f"  Duplicates skipped:  {index_stats.get('duplicates', 0)}")
    log.info(f"  New embeddings:      {emb_stats.get('new_embeddings', 0)}")
    log.info(f"  Time:                {elapsed:.1f}s")
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Daily Telegram Sync')
    parser.add_argument('--setup', action='store_true', help='First time setup')
    parser.add_argument('--hours', type=int, default=36, help='Hours to look back (default: 36)')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch, do not index')
    args = parser.parse_args()

    if args.setup:
        setup_config()
        return

    if args.fetch_only:
        config = load_config()
        if not config:
            log.error("No configuration found. Run: python daily_sync.py --setup")
            sys.exit(1)
        messages = asyncio.run(fetch_messages(config, hours=args.hours))
        # Save to file for inspection
        output = BASE_DIR / 'fetched_messages.json'
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(messages)} messages to {output}")
        return

    run_sync(hours=args.hours, skip_embeddings=args.skip_embeddings)


if __name__ == '__main__':
    main()
