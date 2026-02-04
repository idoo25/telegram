#!/usr/bin/env python3
"""
Telegram Chat Search Utilities
Full-text search and filtering for indexed Telegram messages.

Usage:
    python search.py <query> [options]
    python search.py "שלום" --db telegram.db
    python search.py "link" --user user123 --limit 50
"""

import sqlite3
import argparse
from datetime import datetime
from typing import Optional


class TelegramSearch:
    """Search interface for indexed Telegram messages."""

    def __init__(self, db_path: str = 'telegram.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        from_date: Optional[int] = None,
        to_date: Optional[int] = None,
        has_links: Optional[bool] = None,
        has_mentions: Optional[bool] = None,
        has_media: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Full-text search with optional filters.

        Args:
            query: Search query (supports FTS5 syntax: AND, OR, NOT, "phrase", prefix*)
            user_id: Filter by user ID
            from_date: Filter messages after this Unix timestamp
            to_date: Filter messages before this Unix timestamp
            has_links: Filter messages with/without links
            has_mentions: Filter messages with/without mentions
            has_media: Filter messages with/without media
            limit: Max results to return
            offset: Results offset for pagination

        Returns:
            List of matching messages with relevance score
        """
        # Build the query
        conditions = []
        params = []

        if user_id:
            conditions.append("m.from_id = ?")
            params.append(user_id)

        if from_date:
            conditions.append("m.date_unixtime >= ?")
            params.append(from_date)

        if to_date:
            conditions.append("m.date_unixtime <= ?")
            params.append(to_date)

        if has_links is not None:
            conditions.append("m.has_links = ?")
            params.append(1 if has_links else 0)

        if has_mentions is not None:
            conditions.append("m.has_mentions = ?")
            params.append(1 if has_mentions else 0)

        if has_media is not None:
            conditions.append("m.has_media = ?")
            params.append(1 if has_media else 0)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f'''
            SELECT
                m.id,
                m.date,
                m.date_unixtime,
                m.from_name,
                m.from_id,
                m.text_plain,
                m.reply_to_message_id,
                m.forwarded_from,
                m.has_links,
                m.has_mentions,
                m.has_media,
                bm25(messages_fts) as relevance
            FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.id
            WHERE messages_fts MATCH ?
            AND {where_clause}
            ORDER BY relevance
            LIMIT ? OFFSET ?
        '''

        params = [query] + params + [limit, offset]

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def search_by_user(self, user_id: str, limit: int = 100) -> list[dict]:
        """Get all messages from a specific user."""
        sql = '''
            SELECT * FROM messages
            WHERE from_id = ?
            ORDER BY date_unixtime DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    def search_by_date_range(
        self,
        from_date: int,
        to_date: int,
        limit: int = 1000
    ) -> list[dict]:
        """Get messages within a date range."""
        sql = '''
            SELECT * FROM messages
            WHERE date_unixtime BETWEEN ? AND ?
            ORDER BY date_unixtime ASC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (from_date, to_date, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_thread(self, message_id: int, depth: int = 50) -> list[dict]:
        """
        Get a conversation thread starting from a message.
        Follows reply_to_message_id chain.
        """
        messages = []
        current_id = message_id

        for _ in range(depth):
            cursor = self.conn.execute(
                'SELECT * FROM messages WHERE id = ?',
                (current_id,)
            )
            row = cursor.fetchone()
            if not row:
                break

            messages.append(dict(row))
            if row['reply_to_message_id']:
                current_id = row['reply_to_message_id']
            else:
                break

        return list(reversed(messages))

    def get_replies(self, message_id: int) -> list[dict]:
        """Get all direct replies to a message."""
        sql = '''
            SELECT * FROM messages
            WHERE reply_to_message_id = ?
            ORDER BY date_unixtime ASC
        '''
        cursor = self.conn.execute(sql, (message_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_links(self, limit: int = 100) -> list[dict]:
        """Get all extracted links."""
        sql = '''
            SELECT e.value as url, e.message_id, m.from_name, m.date
            FROM entities e
            JOIN messages m ON e.message_id = m.id
            WHERE e.type = 'link'
            ORDER BY m.date_unixtime DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_mentions(self, username: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Get mentions, optionally filtered by username."""
        if username:
            sql = '''
                SELECT e.value as mention, e.message_id, m.from_name, m.text_plain, m.date
                FROM entities e
                JOIN messages m ON e.message_id = m.id
                WHERE e.type = 'mention' AND e.value LIKE ?
                ORDER BY m.date_unixtime DESC
                LIMIT ?
            '''
            cursor = self.conn.execute(sql, (f'%{username}%', limit))
        else:
            sql = '''
                SELECT e.value as mention, e.message_id, m.from_name, m.text_plain, m.date
                FROM entities e
                JOIN messages m ON e.message_id = m.id
                WHERE e.type = 'mention'
                ORDER BY m.date_unixtime DESC
                LIMIT ?
            '''
            cursor = self.conn.execute(sql, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_forwarded_messages(self, source_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Get forwarded messages, optionally from a specific source."""
        if source_id:
            sql = '''
                SELECT * FROM messages
                WHERE forwarded_from_id = ?
                ORDER BY date_unixtime DESC
                LIMIT ?
            '''
            cursor = self.conn.execute(sql, (source_id, limit))
        else:
            sql = '''
                SELECT * FROM messages
                WHERE forwarded_from IS NOT NULL
                ORDER BY date_unixtime DESC
                LIMIT ?
            '''
            cursor = self.conn.execute(sql, (limit,))

        return [dict(row) for row in cursor.fetchall()]


def format_result(msg: dict) -> str:
    """Format a message for display."""
    date_str = msg.get('date', 'Unknown date')
    from_name = msg.get('from_name', 'Unknown')
    text = msg.get('text_plain', '')[:200]
    if len(msg.get('text_plain', '')) > 200:
        text += '...'

    flags = []
    if msg.get('has_links'):
        flags.append('[link]')
    if msg.get('has_mentions'):
        flags.append('[mention]')
    if msg.get('has_media'):
        flags.append('[media]')

    flags_str = ' '.join(flags)
    return f"[{date_str}] {from_name}: {text} {flags_str}"


def main():
    parser = argparse.ArgumentParser(description='Search indexed Telegram messages')
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--db', default='telegram.db', help='Database path')
    parser.add_argument('--user', help='Filter by user ID')
    parser.add_argument('--from-date', help='From date (YYYY-MM-DD)')
    parser.add_argument('--to-date', help='To date (YYYY-MM-DD)')
    parser.add_argument('--links', action='store_true', help='Show only messages with links')
    parser.add_argument('--mentions', action='store_true', help='Show only messages with mentions')
    parser.add_argument('--media', action='store_true', help='Show only messages with media')
    parser.add_argument('--limit', type=int, default=50, help='Max results')
    parser.add_argument('--list-links', action='store_true', help='List all extracted links')
    parser.add_argument('--list-mentions', action='store_true', help='List all mentions')

    args = parser.parse_args()

    with TelegramSearch(args.db) as search:
        # Special modes
        if args.list_links:
            links = search.get_links(args.limit)
            print(f"Found {len(links)} links:\n")
            for link in links:
                print(f"  {link['url']}")
                print(f"    From: {link['from_name']} at {link['date']}")
            return

        if args.list_mentions:
            mentions = search.get_mentions(limit=args.limit)
            print(f"Found {len(mentions)} mentions:\n")
            for m in mentions:
                print(f"  {m['mention']} by {m['from_name']}")
            return

        if not args.query:
            parser.print_help()
            return

        # Parse dates
        from_ts = None
        to_ts = None
        if args.from_date:
            from_ts = int(datetime.strptime(args.from_date, '%Y-%m-%d').timestamp())
        if args.to_date:
            to_ts = int(datetime.strptime(args.to_date, '%Y-%m-%d').timestamp())

        # Perform search
        results = search.search(
            query=args.query,
            user_id=args.user,
            from_date=from_ts,
            to_date=to_ts,
            has_links=True if args.links else None,
            has_mentions=True if args.mentions else None,
            has_media=True if args.media else None,
            limit=args.limit
        )

        print(f"Found {len(results)} results for '{args.query}':\n")
        for msg in results:
            print(format_result(msg))
            print()


if __name__ == '__main__':
    main()
