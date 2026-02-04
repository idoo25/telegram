#!/usr/bin/env python3
"""
Telegram Chat Search Utilities (Optimized)

Features:
- Full-text search with BM25 ranking
- LRU caching for repeated queries
- Fuzzy search with trigram similarity
- Thread traversal with DFS/BFS
- Autocomplete suggestions

Usage:
    python search.py <query> [options]
    python search.py "שלום" --db telegram.db
    python search.py "link" --user user123 --fuzzy
"""

import sqlite3
import argparse
from datetime import datetime
from typing import Optional
from functools import lru_cache

from data_structures import LRUCache, Trie, TrigramIndex, ReplyGraph, lru_cached


class TelegramSearch:
    """
    High-performance search interface for indexed Telegram messages.

    Features:
    - Full-text search with FTS5 and BM25 ranking
    - Query result caching (LRU)
    - Fuzzy/approximate search with trigrams
    - Thread reconstruction with graph traversal
    - Autocomplete for usernames and common terms
    """

    def __init__(self, db_path: str = 'telegram.db', cache_size: int = 1000):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Initialize caches
        self.query_cache = LRUCache(maxsize=cache_size)
        self.user_trie: Optional[Trie] = None
        self.trigram_index: Optional[TrigramIndex] = None
        self.reply_graph: Optional[ReplyGraph] = None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ==========================================
    # FULL-TEXT SEARCH
    # ==========================================

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
        offset: int = 0,
        use_cache: bool = True
    ) -> list[dict]:
        """
        Full-text search with BM25 ranking and optional filters.

        Args:
            query: FTS5 query (supports AND, OR, NOT, "phrase", prefix*)
            user_id: Filter by user ID
            from_date: Unix timestamp lower bound
            to_date: Unix timestamp upper bound
            has_links/has_mentions/has_media: Boolean filters
            limit: Max results
            offset: Pagination offset
            use_cache: Whether to use LRU cache

        Returns:
            List of message dicts with relevance scores
        """
        # Build cache key
        cache_key = f"search:{query}:{user_id}:{from_date}:{to_date}:{has_links}:{has_mentions}:{has_media}:{limit}:{offset}"

        if use_cache:
            cached = self.query_cache.get(cache_key)
            if cached is not None:
                return cached

        # Build query conditions
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
                bm25(messages_fts, 1.0, 0.5) as relevance
            FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.id
            WHERE messages_fts MATCH ?
            AND {where_clause}
            ORDER BY relevance
            LIMIT ? OFFSET ?
        '''

        params = [query] + params + [limit, offset]

        cursor = self.conn.execute(sql, params)
        results = [dict(row) for row in cursor.fetchall()]

        if use_cache:
            self.query_cache.put(cache_key, results)

        return results

    def search_prefix(self, prefix: str, limit: int = 100) -> list[dict]:
        """
        Search using prefix matching (autocomplete-style).

        Uses FTS5 prefix index for fast prefix queries.
        """
        # FTS5 prefix search syntax
        query = f'{prefix}*'
        return self.search(query, limit=limit, use_cache=True)

    # ==========================================
    # FUZZY SEARCH
    # ==========================================

    def fuzzy_search(
        self,
        query: str,
        threshold: float = 0.3,
        limit: int = 50
    ) -> list[dict]:
        """
        Fuzzy search using trigram similarity.

        Finds messages even with typos or slight variations.

        Args:
            query: Search query
            threshold: Minimum similarity (0-1)
            limit: Max results

        Returns:
            List of (message, similarity) tuples
        """
        # Build trigram index if not exists
        if self.trigram_index is None:
            self._build_trigram_index()

        # Search trigram index
        matches = self.trigram_index.search(query, threshold=threshold, limit=limit)

        # Fetch full messages
        results = []
        for msg_id, similarity in matches:
            cursor = self.conn.execute(
                'SELECT * FROM messages WHERE id = ?',
                (msg_id,)
            )
            row = cursor.fetchone()
            if row:
                msg = dict(row)
                msg['similarity'] = similarity
                results.append(msg)

        return results

    def _build_trigram_index(self) -> None:
        """Build in-memory trigram index from database."""
        print("Building trigram index (first time only)...")
        self.trigram_index = TrigramIndex()

        cursor = self.conn.execute(
            'SELECT id, text_plain FROM messages WHERE text_plain IS NOT NULL'
        )
        for row in cursor.fetchall():
            self.trigram_index.add(row[0], row[1])

        print(f"Trigram index built: {len(self.trigram_index)} documents")

    # ==========================================
    # THREAD TRAVERSAL
    # ==========================================

    def get_thread_dfs(self, message_id: int) -> list[dict]:
        """
        Get full conversation thread using DFS traversal.

        Returns messages in depth-first order (follows reply chains deep).
        """
        if self.reply_graph is None:
            self._build_reply_graph()

        # Find thread root
        root_id = self.reply_graph.get_thread_root(message_id)

        # DFS traversal
        msg_ids = self.reply_graph.dfs_descendants(root_id)

        # Fetch messages in order
        return self._fetch_messages_ordered(msg_ids)

    def get_thread_bfs(self, message_id: int) -> list[dict]:
        """
        Get conversation thread using BFS traversal.

        Returns messages level by level.
        """
        if self.reply_graph is None:
            self._build_reply_graph()

        root_id = self.reply_graph.get_thread_root(message_id)
        msg_ids = self.reply_graph.bfs_descendants(root_id)

        return self._fetch_messages_ordered(msg_ids)

    def get_thread_with_depth(self, message_id: int) -> list[tuple[dict, int]]:
        """
        Get thread with depth information for each message.

        Returns list of (message, depth) tuples.
        """
        if self.reply_graph is None:
            self._build_reply_graph()

        root_id = self.reply_graph.get_thread_root(message_id)
        items = self.reply_graph.bfs_with_depth(root_id)

        results = []
        for msg_id, depth in items:
            cursor = self.conn.execute(
                'SELECT * FROM messages WHERE id = ?',
                (msg_id,)
            )
            row = cursor.fetchone()
            if row:
                results.append((dict(row), depth))

        return results

    def get_replies(self, message_id: int) -> list[dict]:
        """Get all direct replies to a message."""
        if self.reply_graph is None:
            self._build_reply_graph()

        child_ids = self.reply_graph.get_children(message_id)
        return self._fetch_messages_ordered(child_ids)

    def get_conversation_path(self, message_id: int) -> list[dict]:
        """Get the path from thread root to this message."""
        if self.reply_graph is None:
            self._build_reply_graph()

        path_ids = self.reply_graph.get_thread_path(message_id)
        return self._fetch_messages_ordered(path_ids)

    def _build_reply_graph(self) -> None:
        """Build in-memory reply graph from database."""
        print("Building reply graph (first time only)...")
        self.reply_graph = ReplyGraph()

        cursor = self.conn.execute(
            'SELECT id, reply_to_message_id FROM messages'
        )
        for row in cursor.fetchall():
            self.reply_graph.add_message(row[0], row[1])

        print(f"Reply graph built: {self.reply_graph.stats}")

    def _fetch_messages_ordered(self, msg_ids: list[int]) -> list[dict]:
        """Fetch messages preserving the order of IDs."""
        if not msg_ids:
            return []

        placeholders = ','.join('?' * len(msg_ids))
        cursor = self.conn.execute(
            f'SELECT * FROM messages WHERE id IN ({placeholders})',
            msg_ids
        )

        # Create lookup dict
        msg_map = {row['id']: dict(row) for row in cursor.fetchall()}

        # Return in original order
        return [msg_map[mid] for mid in msg_ids if mid in msg_map]

    # ==========================================
    # AUTOCOMPLETE
    # ==========================================

    def autocomplete_user(self, prefix: str, limit: int = 10) -> list[str]:
        """
        Autocomplete username suggestions.

        Uses Trie for O(p + k) lookup where p=prefix length, k=results.
        """
        if self.user_trie is None:
            self._build_user_trie()

        return self.user_trie.autocomplete(prefix, limit=limit)

    def _build_user_trie(self) -> None:
        """Build Trie index for usernames."""
        self.user_trie = Trie()

        cursor = self.conn.execute('SELECT user_id, display_name FROM users')
        for row in cursor.fetchall():
            if row['display_name']:
                self.user_trie.insert(row['display_name'], data=row['user_id'])
            if row['user_id']:
                self.user_trie.insert(row['user_id'], data=row['user_id'])

    # ==========================================
    # CONVENIENCE METHODS
    # ==========================================

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

    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.query_cache.stats


def format_result(msg: dict, show_depth: bool = False, depth: int = 0) -> str:
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
    if msg.get('similarity'):
        flags.append(f'[sim:{msg["similarity"]:.2f}]')
    if msg.get('relevance'):
        flags.append(f'[rel:{abs(msg["relevance"]):.2f}]')

    flags_str = ' '.join(flags)
    indent = '  ' * depth if show_depth else ''
    return f"{indent}[{date_str}] {from_name}: {text} {flags_str}"


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
    parser.add_argument('--fuzzy', action='store_true', help='Use fuzzy search')
    parser.add_argument('--threshold', type=float, default=0.3, help='Fuzzy match threshold')
    parser.add_argument('--thread', type=int, help='Show thread for message ID')
    parser.add_argument('--list-links', action='store_true', help='List all extracted links')
    parser.add_argument('--list-mentions', action='store_true', help='List all mentions')
    parser.add_argument('--autocomplete', help='Autocomplete username')
    parser.add_argument('--cache-stats', action='store_true', help='Show cache statistics')

    args = parser.parse_args()

    with TelegramSearch(args.db) as search:
        # Show thread
        if args.thread:
            print(f"Thread containing message {args.thread}:\n")
            thread = search.get_thread_with_depth(args.thread)
            for msg, depth in thread:
                print(format_result(msg, show_depth=True, depth=depth))
            return

        # Autocomplete
        if args.autocomplete:
            suggestions = search.autocomplete_user(args.autocomplete)
            print(f"Suggestions for '{args.autocomplete}':")
            for s in suggestions:
                print(f"  {s}")
            return

        # List links
        if args.list_links:
            links = search.get_links(args.limit)
            print(f"Found {len(links)} links:\n")
            for link in links:
                print(f"  {link['url']}")
                print(f"    From: {link['from_name']} at {link['date']}")
            return

        # List mentions
        if args.list_mentions:
            mentions = search.get_mentions(limit=args.limit)
            print(f"Found {len(mentions)} mentions:\n")
            for m in mentions:
                print(f"  {m['mention']} by {m['from_name']}")
            return

        # Cache stats
        if args.cache_stats:
            print(f"Cache stats: {search.cache_stats}")
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

        # Fuzzy or regular search
        if args.fuzzy:
            results = search.fuzzy_search(
                query=args.query,
                threshold=args.threshold,
                limit=args.limit
            )
            print(f"Found {len(results)} fuzzy matches for '{args.query}':\n")
        else:
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

        # Show cache stats
        print(f"\nCache: {search.cache_stats}")


if __name__ == '__main__':
    main()
