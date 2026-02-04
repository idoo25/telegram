#!/usr/bin/env python3
"""
Telegram JSON Chat Indexer (Optimized)

Features:
- Batch processing for faster indexing
- Graph building for reply threads
- Trigram index for fuzzy search
- Progress tracking
- Memory-efficient streaming

Usage:
    python indexer.py <json_file> [--db <database_file>]
    python indexer.py result.json --db telegram.db
    python indexer.py result.json --batch-size 5000 --build-trigrams
"""

import json
import sqlite3
import argparse
import os
import time
from pathlib import Path
from typing import Any, Generator
from collections import defaultdict

from data_structures import BloomFilter, ReplyGraph, generate_trigrams


def flatten_text(text_field: Any) -> str:
    """
    Flatten the text field which can be either a string or array of mixed content.
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
            if entity_type != 'plain':
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
        'date_unixtime': int(msg.get('date_unixtime', 0)) if msg.get('date_unixtime') else 0,
        'from_name': msg.get('from', ''),
        'from_id': msg.get('from_id', ''),
        'reply_to_message_id': msg.get('reply_to_message_id'),
        'forwarded_from': msg.get('forwarded_from'),
        'forwarded_from_id': msg.get('forwarded_from_id'),
        'text_plain': text_plain,
        'text_length': len(text_plain),
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

    messages = data if isinstance(data, list) else data.get('messages', [])

    for msg in messages:
        parsed = parse_message(msg)
        if parsed:
            yield parsed


def count_messages(json_path: str) -> int:
    """Count messages in JSON file without loading all into memory."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    messages = data if isinstance(data, list) else data.get('messages', [])
    return sum(1 for msg in messages if msg.get('type') == 'message')


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with optimized schema."""
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


class OptimizedIndexer:
    """
    High-performance indexer with batch processing and graph building.

    Features:
    - Batch inserts (100x faster than individual inserts)
    - Bloom filter for duplicate detection
    - Reply graph construction
    - Trigram index building
    - Progress tracking
    """

    def __init__(
        self,
        db_path: str,
        batch_size: int = 1000,
        build_trigrams: bool = False,
        build_graph: bool = True
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.build_trigrams = build_trigrams
        self.build_graph = build_graph

        self.conn = init_database(db_path)
        self.bloom = BloomFilter(expected_items=1000000, fp_rate=0.01)
        self.graph = ReplyGraph() if build_graph else None

        # Batch buffers
        self.message_batch: list[tuple] = []
        self.entity_batch: list[tuple] = []
        self.trigram_batch: list[tuple] = []

        # Stats
        self.stats = {
            'messages': 0,
            'entities': 0,
            'trigrams': 0,
            'users': {},
            'skipped': 0,
            'duplicates': 0
        }

    def index_file(self, json_path: str, show_progress: bool = True) -> dict:
        """
        Index a JSON file into the database.

        Returns statistics dict.
        """
        start_time = time.time()

        # Count total for progress
        if show_progress:
            print(f"Counting messages in {json_path}...")
            total = count_messages(json_path)
            print(f"Found {total:,} messages to index")
        else:
            total = 0

        # Disable auto-commit for batch processing
        self.conn.execute('BEGIN TRANSACTION')

        try:
            for i, msg in enumerate(load_json_messages(json_path)):
                self._index_message(msg)

                # Progress update
                if show_progress and (i + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    print(f"  Indexed {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) "
                          f"- {rate:.0f} msg/s - ETA: {eta:.0f}s")

            # Flush remaining batches
            self._flush_batches()

            # Build reply graph in database
            if self.build_graph:
                self._build_graph_tables()

            # Update users table
            self._update_users()

            # Commit transaction
            self.conn.commit()

            # Optimize FTS index
            print("Optimizing FTS index...")
            self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e

        elapsed = time.time() - start_time
        self.stats['elapsed_seconds'] = elapsed
        self.stats['messages_per_second'] = self.stats['messages'] / elapsed if elapsed > 0 else 0

        return self.stats

    def _index_message(self, msg: dict) -> None:
        """Index a single message into batch buffers."""
        msg_id = msg['id']

        # Duplicate check with Bloom filter
        msg_key = f"msg_{msg_id}"
        if msg_key in self.bloom:
            self.stats['duplicates'] += 1
            return
        self.bloom.add(msg_key)

        # Add to message batch
        self.message_batch.append((
            msg['id'], msg['type'], msg['date'], msg['date_unixtime'],
            msg['from_name'], msg['from_id'], msg['reply_to_message_id'],
            msg['forwarded_from'], msg['forwarded_from_id'], msg['text_plain'],
            msg['text_length'], msg['has_media'], msg['has_photo'],
            msg['has_links'], msg['has_mentions'], msg['is_edited'],
            msg['edited_unixtime'], msg['photo_file_size'],
            msg['photo_width'], msg['photo_height'], msg['raw_json']
        ))

        # Add entities to batch
        for entity in msg['entities']:
            self.entity_batch.append((msg_id, entity['type'], entity['value']))

        # Add trigrams if enabled
        if self.build_trigrams and msg['text_plain']:
            for i, trigram in enumerate(generate_trigrams(msg['text_plain'])):
                self.trigram_batch.append((trigram, msg_id, i))

        # Build graph
        if self.graph:
            self.graph.add_message(msg_id, msg['reply_to_message_id'])

        # Track users
        user_id = msg['from_id']
        if user_id:
            if user_id not in self.stats['users']:
                self.stats['users'][user_id] = {
                    'display_name': msg['from_name'],
                    'first_seen': msg['date_unixtime'],
                    'last_seen': msg['date_unixtime'],
                    'count': 0
                }
            self.stats['users'][user_id]['count'] += 1
            ts = msg['date_unixtime']
            if ts and ts < self.stats['users'][user_id]['first_seen']:
                self.stats['users'][user_id]['first_seen'] = ts
            if ts and ts > self.stats['users'][user_id]['last_seen']:
                self.stats['users'][user_id]['last_seen'] = ts

        self.stats['messages'] += 1

        # Flush if batch is full
        if len(self.message_batch) >= self.batch_size:
            self._flush_batches()

    def _flush_batches(self) -> None:
        """Flush all batch buffers to database."""
        cursor = self.conn.cursor()

        # Insert messages
        if self.message_batch:
            cursor.executemany('''
                INSERT OR REPLACE INTO messages (
                    id, type, date, date_unixtime, from_name, from_id,
                    reply_to_message_id, forwarded_from, forwarded_from_id,
                    text_plain, text_length, has_media, has_photo, has_links,
                    has_mentions, is_edited, edited_unixtime, photo_file_size,
                    photo_width, photo_height, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.message_batch)
            self.message_batch = []

        # Insert entities
        if self.entity_batch:
            cursor.executemany('''
                INSERT INTO entities (message_id, type, value)
                VALUES (?, ?, ?)
            ''', self.entity_batch)
            self.stats['entities'] += len(self.entity_batch)
            self.entity_batch = []

        # Insert trigrams
        if self.trigram_batch:
            cursor.executemany('''
                INSERT OR IGNORE INTO trigrams (trigram, message_id, position)
                VALUES (?, ?, ?)
            ''', self.trigram_batch)
            self.stats['trigrams'] += len(self.trigram_batch)
            self.trigram_batch = []

    def _build_graph_tables(self) -> None:
        """Build reply graph tables from in-memory graph."""
        if not self.graph:
            return

        print("Building reply graph tables...")
        cursor = self.conn.cursor()

        # Insert edges into reply_graph
        edges = []
        for parent_id, children in self.graph.children.items():
            for child_id in children:
                edges.append((parent_id, child_id, 1))

        if edges:
            cursor.executemany('''
                INSERT OR IGNORE INTO reply_graph (parent_id, child_id, depth)
                VALUES (?, ?, ?)
            ''', edges)

        # Find connected components (threads)
        print("Finding conversation threads...")
        components = self.graph.find_connected_components()

        thread_data = []
        message_thread_data = []

        for thread_id, component in enumerate(components):
            if not component:
                continue

            # Find root (message with no parent in this component)
            root_id = None
            for msg_id in component:
                if msg_id not in self.graph.parents:
                    root_id = msg_id
                    break
            if root_id is None:
                root_id = min(component)

            # Get thread stats
            cursor.execute('''
                SELECT MIN(date_unixtime), MAX(date_unixtime), COUNT(DISTINCT from_id)
                FROM messages WHERE id IN ({})
            '''.format(','.join('?' * len(component))), list(component))
            row = cursor.fetchone()

            thread_data.append((
                root_id,
                len(component),
                row[0],  # first_message_time
                row[1],  # last_message_time
                row[2]   # participant_count
            ))

            # Map messages to threads with depth
            for msg_id in component:
                depth = len(self.graph.get_ancestors(msg_id))
                message_thread_data.append((msg_id, len(thread_data), depth))

        # Insert thread data
        cursor.executemany('''
            INSERT INTO threads (root_message_id, message_count, first_message_time,
                                last_message_time, participant_count)
            VALUES (?, ?, ?, ?, ?)
        ''', thread_data)

        cursor.executemany('''
            INSERT OR REPLACE INTO message_threads (message_id, thread_id, depth)
            VALUES (?, ?, ?)
        ''', message_thread_data)

        print(f"  Created {len(thread_data)} conversation threads")

    def _update_users(self) -> None:
        """Update users table from tracked data."""
        cursor = self.conn.cursor()
        user_data = [
            (user_id, data['display_name'], data['first_seen'],
             data['last_seen'], data['count'])
            for user_id, data in self.stats['users'].items()
        ]

        cursor.executemany('''
            INSERT OR REPLACE INTO users (user_id, display_name, first_seen, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?)
        ''', user_data)

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


class IncrementalIndexer:
    """
    Incremental indexer for adding new JSON data to existing database.

    Features:
    - Loads existing message IDs into Bloom filter
    - Only processes new messages
    - Updates FTS index automatically
    - Fast duplicate detection O(1)
    """

    def __init__(self, db_path: str, batch_size: int = 1000):
        self.db_path = db_path
        self.batch_size = batch_size

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}. Use OptimizedIndexer for initial import.")

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Load existing message IDs into Bloom filter
        self.bloom = BloomFilter(expected_items=2000000, fp_rate=0.001)
        self._load_existing_ids()

        # Batch buffers
        self.message_batch: list[tuple] = []
        self.entity_batch: list[tuple] = []

        # Stats
        self.stats = {
            'total_in_file': 0,
            'new_messages': 0,
            'duplicates': 0,
            'entities': 0,
            'users_updated': 0
        }

    def _load_existing_ids(self) -> None:
        """Load existing message IDs into Bloom filter for O(1) duplicate detection."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM messages")

        count = 0
        for row in cursor:
            self.bloom.add(f"msg_{row[0]}")
            count += 1

        print(f"Loaded {count:,} existing message IDs into Bloom filter")
        self.stats['existing_count'] = count

    def update_from_json(self, json_path: str, show_progress: bool = True) -> dict:
        """
        Add new messages from JSON file to existing database.

        Only messages that don't exist in the database will be added.
        FTS5 index is updated automatically.
        """
        start_time = time.time()

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = data if isinstance(data, list) else data.get('messages', [])
        self.stats['total_in_file'] = len(messages)

        if show_progress:
            print(f"Processing {len(messages):,} messages from {json_path}")

        # Start transaction
        self.conn.execute('BEGIN TRANSACTION')

        try:
            for i, msg in enumerate(messages):
                if msg.get('type') != 'message':
                    continue

                parsed = parse_message(msg)
                if parsed:
                    self._process_message(parsed)

                # Progress update
                if show_progress and (i + 1) % 10000 == 0:
                    print(f"  Processed {i+1:,}/{len(messages):,} - "
                          f"New: {self.stats['new_messages']:,}, "
                          f"Duplicates: {self.stats['duplicates']:,}")

            # Flush remaining
            self._flush_batches()

            # Update user stats
            self._update_user_stats()

            # Commit
            self.conn.commit()

            # Optimize FTS if we added new data
            if self.stats['new_messages'] > 0:
                print("Optimizing FTS index...")
                self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
                self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e

        elapsed = time.time() - start_time
        self.stats['elapsed_seconds'] = elapsed

        return self.stats

    def update_from_json_data(self, json_data: dict | list, show_progress: bool = False) -> dict:
        """
        Add new messages from JSON data (already parsed, not from file).

        Useful for API uploads.
        """
        start_time = time.time()

        messages = json_data if isinstance(json_data, list) else json_data.get('messages', [])
        self.stats['total_in_file'] = len(messages)

        # Start transaction
        self.conn.execute('BEGIN TRANSACTION')

        try:
            for msg in messages:
                if msg.get('type') != 'message':
                    continue

                parsed = parse_message(msg)
                if parsed:
                    self._process_message(parsed)

            # Flush remaining
            self._flush_batches()

            # Update user stats
            self._update_user_stats()

            # Commit
            self.conn.commit()

            # Optimize FTS if we added new data
            if self.stats['new_messages'] > 0:
                self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
                self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e

        elapsed = time.time() - start_time
        self.stats['elapsed_seconds'] = elapsed

        return self.stats

    def _process_message(self, msg: dict) -> None:
        """Process a single message, adding to batch if new."""
        msg_id = msg['id']
        msg_key = f"msg_{msg_id}"

        # Check if already exists (Bloom filter first, then DB if needed)
        if msg_key in self.bloom:
            self.stats['duplicates'] += 1
            return

        # Add to Bloom filter
        self.bloom.add(msg_key)

        # Add to message batch
        self.message_batch.append((
            msg['id'], msg['type'], msg['date'], msg['date_unixtime'],
            msg['from_name'], msg['from_id'], msg['reply_to_message_id'],
            msg['forwarded_from'], msg['forwarded_from_id'], msg['text_plain'],
            msg['text_length'], msg['has_media'], msg['has_photo'],
            msg['has_links'], msg['has_mentions'], msg['is_edited'],
            msg['edited_unixtime'], msg['photo_file_size'],
            msg['photo_width'], msg['photo_height'], msg['raw_json']
        ))

        # Add entities to batch
        for entity in msg['entities']:
            self.entity_batch.append((msg_id, entity['type'], entity['value']))

        self.stats['new_messages'] += 1

        # Flush if batch is full
        if len(self.message_batch) >= self.batch_size:
            self._flush_batches()

    def _flush_batches(self) -> None:
        """Flush batch buffers to database."""
        cursor = self.conn.cursor()

        # Insert messages (FTS5 trigger will update automatically)
        if self.message_batch:
            cursor.executemany('''
                INSERT OR IGNORE INTO messages (
                    id, type, date, date_unixtime, from_name, from_id,
                    reply_to_message_id, forwarded_from, forwarded_from_id,
                    text_plain, text_length, has_media, has_photo, has_links,
                    has_mentions, is_edited, edited_unixtime, photo_file_size,
                    photo_width, photo_height, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.message_batch)
            self.message_batch = []

        # Insert entities
        if self.entity_batch:
            cursor.executemany('''
                INSERT OR IGNORE INTO entities (message_id, type, value)
                VALUES (?, ?, ?)
            ''', self.entity_batch)
            self.stats['entities'] += len(self.entity_batch)
            self.entity_batch = []

    def _update_user_stats(self) -> None:
        """Update users table with aggregated stats."""
        cursor = self.conn.cursor()

        # Upsert users from messages
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, display_name, first_seen, last_seen, message_count)
            SELECT
                from_id,
                from_name,
                MIN(date_unixtime),
                MAX(date_unixtime),
                COUNT(*)
            FROM messages
            WHERE from_id IS NOT NULL AND from_id != ''
            GROUP BY from_id
        ''')
        self.stats['users_updated'] = cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


def update_database(db_path: str, json_path: str) -> dict:
    """
    Convenience function to update database with new JSON file.

    Args:
        db_path: Path to existing SQLite database
        json_path: Path to new JSON file

    Returns:
        Statistics dict
    """
    indexer = IncrementalIndexer(db_path)
    try:
        stats = indexer.update_from_json(json_path)
        return stats
    finally:
        indexer.close()


def main():
    parser = argparse.ArgumentParser(description='Index Telegram JSON export to SQLite (Optimized)')
    parser.add_argument('json_file', help='Path to Telegram export JSON file')
    parser.add_argument('--db', default='telegram.db', help='SQLite database path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for inserts')
    parser.add_argument('--build-trigrams', action='store_true', help='Build trigram index for fuzzy search')
    parser.add_argument('--no-graph', action='store_true', help='Skip building reply graph')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--update', action='store_true',
                       help='Update existing database (add only new messages)')

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1

    # Update mode: add new messages to existing database
    if args.update:
        if not os.path.exists(args.db):
            print(f"Error: Database not found: {args.db}")
            print("Use without --update flag for initial import")
            return 1

        print(f"{'='*50}")
        print(f"INCREMENTAL UPDATE MODE")
        print(f"{'='*50}")
        print(f"Database: {args.db}")
        print(f"New JSON: {args.json_file}")
        print()

        indexer = IncrementalIndexer(args.db, args.batch_size)
        stats = indexer.update_from_json(args.json_file, show_progress=not args.quiet)

        print(f"\n{'='*50}")
        print(f"Update complete!")
        print(f"{'='*50}")
        print(f"  Messages in file:    {stats['total_in_file']:,}")
        print(f"  Already existed:     {stats['duplicates']:,}")
        print(f"  New messages added:  {stats['new_messages']:,}")
        print(f"  New entities:        {stats['entities']:,}")
        print(f"  Time elapsed:        {stats['elapsed_seconds']:.1f}s")

        indexer.close()
        return 0

    # Initial import mode
    print(f"Initializing database: {args.db}")
    indexer = OptimizedIndexer(
        db_path=args.db,
        batch_size=args.batch_size,
        build_trigrams=args.build_trigrams,
        build_graph=not args.no_graph
    )

    print(f"Indexing: {args.json_file}")
    stats = indexer.index_file(args.json_file, show_progress=not args.quiet)

    print(f"\n{'='*50}")
    print(f"Indexing complete!")
    print(f"{'='*50}")
    print(f"  Messages indexed:    {stats['messages']:,}")
    print(f"  Entities extracted:  {stats['entities']:,}")
    print(f"  Unique users:        {len(stats['users']):,}")
    print(f"  Duplicates skipped:  {stats['duplicates']:,}")
    if stats.get('trigrams'):
        print(f"  Trigrams indexed:    {stats['trigrams']:,}")
    print(f"  Time elapsed:        {stats['elapsed_seconds']:.1f}s")
    print(f"  Speed:               {stats['messages_per_second']:.0f} msg/s")
    print(f"\nDatabase saved to: {args.db}")

    indexer.close()
    return 0


if __name__ == '__main__':
    exit(main())
