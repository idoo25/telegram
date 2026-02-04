#!/usr/bin/env python3
"""
Telegram Chat Analytics (Enhanced with Course Algorithms)

Features:
- LCS-based similar message detection
- Heap-based Top-K (O(n log k) instead of O(n log n))
- Selection algorithm for O(n) median/percentiles
- Rank Tree for order statistics queries
- Bucket Sort for time-based histograms

Usage:
    python analyzer.py --db telegram.db [options]
    python analyzer.py --stats
    python analyzer.py --top-users
    python analyzer.py --similar        # NEW: Find similar messages
    python analyzer.py --percentiles    # NEW: Message length percentiles
    python analyzer.py --user-rank USER # NEW: Get user's rank
"""

import sqlite3
import argparse
import json
from collections import Counter
from datetime import datetime
from typing import Optional
import re

# Import course algorithms
from algorithms import (
    # LCS
    lcs_similarity, find_similar_messages,
    # Top-K
    TopK, top_k_frequent, top_k_by_field,
    # Selection
    find_median, find_percentile,
    # Rank Tree
    RankTree,
    # Bucket Sort
    bucket_sort_by_time, time_histogram, hourly_distribution,
    # Combined
    RankedTimeIndex
)


class TelegramAnalyzer:
    """
    Analytics interface for indexed Telegram messages.

    Enhanced with efficient algorithms:
    - Top-K queries: O(n log k) using heap
    - Percentiles: O(n) using selection algorithm
    - Rank queries: O(log n) using rank tree
    - Similar messages: LCS-based detection
    """

    def __init__(self, db_path: str = 'telegram.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Lazy-loaded data structures
        self._rank_tree: Optional[RankTree] = None
        self._time_index: Optional[RankedTimeIndex] = None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ==========================================
    # ORIGINAL METHODS (kept for compatibility)
    # ==========================================

    def get_stats(self) -> dict:
        """Get general statistics about the indexed data."""
        stats = {}

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages')
        stats['total_messages'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(DISTINCT from_id) FROM messages')
        stats['total_users'] = cursor.fetchone()[0]

        cursor = self.conn.execute('''
            SELECT MIN(date_unixtime), MAX(date_unixtime) FROM messages
            WHERE date_unixtime IS NOT NULL
        ''')
        row = cursor.fetchone()
        if row[0] and row[1]:
            stats['first_message'] = datetime.fromtimestamp(row[0]).isoformat()
            stats['last_message'] = datetime.fromtimestamp(row[1]).isoformat()
            stats['days_span'] = (row[1] - row[0]) // 86400

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_media = 1')
        stats['messages_with_media'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_links = 1')
        stats['messages_with_links'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_mentions = 1')
        stats['messages_with_mentions'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE forwarded_from IS NOT NULL')
        stats['forwarded_messages'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE reply_to_message_id IS NOT NULL')
        stats['reply_messages'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE is_edited = 1')
        stats['edited_messages'] = cursor.fetchone()[0]

        cursor = self.conn.execute('SELECT type, COUNT(*) FROM entities GROUP BY type')
        stats['entities'] = {row[0]: row[1] for row in cursor.fetchall()}

        # NEW: Add percentile stats using Selection algorithm
        lengths = self._get_message_lengths()
        if lengths:
            stats['median_message_length'] = find_median(lengths)
            stats['p90_message_length'] = find_percentile(lengths, 90)

        return stats

    def _get_message_lengths(self) -> list[int]:
        """Get all message lengths for statistical analysis."""
        cursor = self.conn.execute(
            'SELECT length(text_plain) FROM messages WHERE text_plain IS NOT NULL'
        )
        return [row[0] for row in cursor.fetchall() if row[0]]

    # ==========================================
    # ENHANCED TOP-K METHODS (using Heap)
    # ==========================================

    def get_top_users(self, limit: int = 20) -> list[dict]:
        """
        Get most active users by message count.

        Uses Heap-based Top-K: O(n log k) instead of O(n log n)
        """
        cursor = self.conn.execute('''
            SELECT
                from_id,
                from_name,
                COUNT(*) as message_count,
                SUM(has_links) as links_shared,
                SUM(has_media) as media_shared,
                MIN(date_unixtime) as first_message,
                MAX(date_unixtime) as last_message
            FROM messages
            WHERE from_id IS NOT NULL AND from_id != ''
            GROUP BY from_id
        ''')

        # Use heap-based Top-K
        top = TopK(limit, key=lambda x: x['message_count'])
        for row in cursor.fetchall():
            top.push(dict(row))

        return top.get_top()

    def get_top_words_heap(self, limit: int = 50, min_length: int = 3) -> list[tuple[str, int]]:
        """
        Get most frequent words using Heap-based Top-K.

        O(n + m log k) where n=total words, m=unique words, k=limit
        """
        cursor = self.conn.execute('SELECT text_plain FROM messages WHERE text_plain IS NOT NULL')

        word_pattern = re.compile(r'[\u0590-\u05FFa-zA-Z]+')
        words = []

        for row in cursor.fetchall():
            text = row[0]
            for word in word_pattern.findall(text.lower()):
                if len(word) >= min_length:
                    words.append(word)

        return top_k_frequent(words, limit)

    def get_top_domains_heap(self, limit: int = 20) -> list[tuple[str, int]]:
        """Get most shared domains using Heap-based Top-K."""
        cursor = self.conn.execute("SELECT value FROM entities WHERE type = 'link'")

        domain_pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
        domains = []

        for row in cursor.fetchall():
            match = domain_pattern.match(row[0])
            if match:
                domains.append(match.group(1))

        return top_k_frequent(domains, limit)

    # ==========================================
    # LCS-BASED SIMILAR MESSAGE DETECTION
    # ==========================================

    def find_similar_messages(
        self,
        threshold: float = 0.7,
        min_length: int = 30,
        limit: int = 100,
        sample_size: int = 1000
    ) -> list[tuple[int, int, float, str, str]]:
        """
        Find similar/duplicate messages using LCS algorithm.

        Args:
            threshold: Minimum similarity (0-1)
            min_length: Minimum message length to consider
            limit: Maximum pairs to return
            sample_size: Sample size for large datasets

        Returns:
            List of (id1, id2, similarity, text1, text2) tuples
        """
        cursor = self.conn.execute('''
            SELECT id, text_plain FROM messages
            WHERE text_plain IS NOT NULL AND length(text_plain) >= ?
            ORDER BY RANDOM()
            LIMIT ?
        ''', (min_length, sample_size))

        messages = [(row[0], row[1]) for row in cursor.fetchall()]

        # Find similar pairs using LCS
        similar_pairs = find_similar_messages(messages, threshold, min_length)

        # Fetch full text for results
        results = []
        for id1, id2, sim in similar_pairs[:limit]:
            cursor = self.conn.execute(
                'SELECT text_plain FROM messages WHERE id IN (?, ?)',
                (id1, id2)
            )
            rows = cursor.fetchall()
            if len(rows) == 2:
                results.append((id1, id2, sim, rows[0][0][:100], rows[1][0][:100]))

        return results

    def find_reposts(self, threshold: float = 0.9) -> list[dict]:
        """
        Find potential reposts (very similar messages from different users).
        """
        cursor = self.conn.execute('''
            SELECT id, from_id, text_plain FROM messages
            WHERE text_plain IS NOT NULL AND length(text_plain) >= 50
            ORDER BY date_unixtime DESC
            LIMIT 500
        ''')

        messages = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
        reposts = []

        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                id1, user1, text1 = messages[i]
                id2, user2, text2 = messages[j]

                # Only consider different users
                if user1 == user2:
                    continue

                sim = lcs_similarity(text1, text2)
                if sim >= threshold:
                    reposts.append({
                        'message_id_1': id1,
                        'message_id_2': id2,
                        'user_1': user1,
                        'user_2': user2,
                        'similarity': sim,
                        'text_preview': text1[:80]
                    })

        return sorted(reposts, key=lambda x: x['similarity'], reverse=True)

    # ==========================================
    # SELECTION ALGORITHM (PERCENTILES)
    # ==========================================

    def get_message_length_stats(self) -> dict:
        """
        Get message length statistics using O(n) Selection algorithm.

        Much faster than sorting for percentile calculations.
        """
        lengths = self._get_message_lengths()

        if not lengths:
            return {}

        return {
            'count': len(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'median': find_median(lengths),
            'p25': find_percentile(lengths, 25),
            'p75': find_percentile(lengths, 75),
            'p90': find_percentile(lengths, 90),
            'p95': find_percentile(lengths, 95),
            'p99': find_percentile(lengths, 99),
        }

    def get_response_time_percentiles(self) -> dict:
        """
        Calculate response time percentiles for replies.

        Uses Selection algorithm for O(n) percentile calculation.
        """
        cursor = self.conn.execute('''
            SELECT
                m1.date_unixtime - m2.date_unixtime as response_time
            FROM messages m1
            JOIN messages m2 ON m1.reply_to_message_id = m2.id
            WHERE m1.date_unixtime > m2.date_unixtime
        ''')

        times = [row[0] for row in cursor.fetchall() if row[0] and row[0] > 0]

        if not times:
            return {}

        return {
            'count': len(times),
            'median_seconds': find_median(times),
            'p75_seconds': find_percentile(times, 75),
            'p90_seconds': find_percentile(times, 90),
            'p95_seconds': find_percentile(times, 95),
        }

    # ==========================================
    # RANK TREE (ORDER STATISTICS)
    # ==========================================

    def _build_user_rank_tree(self) -> RankTree:
        """Build rank tree for user activity ranking."""
        if self._rank_tree is not None:
            return self._rank_tree

        self._rank_tree = RankTree()

        cursor = self.conn.execute('''
            SELECT from_id, from_name, COUNT(*) as msg_count
            FROM messages
            WHERE from_id IS NOT NULL AND from_id != ''
            GROUP BY from_id
        ''')

        for row in cursor.fetchall():
            self._rank_tree.insert(
                row['msg_count'],
                {'user_id': row['from_id'], 'name': row['from_name'], 'count': row['msg_count']}
            )

        return self._rank_tree

    def get_user_rank(self, user_id: str) -> dict:
        """
        Get a user's rank among all users.

        Uses Rank Tree: O(log n) instead of O(n log n)
        """
        tree = self._build_user_rank_tree()

        # Get user's message count
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM messages WHERE from_id = ?',
            (user_id,)
        )
        count = cursor.fetchone()[0]

        if count == 0:
            return {'error': 'User not found'}

        rank = tree.rank(count)
        total_users = len(tree)

        return {
            'user_id': user_id,
            'message_count': count,
            'rank': total_users - rank + 1,  # Reverse for "top" ranking
            'total_users': total_users,
            'percentile': ((total_users - rank) / total_users) * 100
        }

    def get_user_by_rank(self, rank: int) -> Optional[dict]:
        """
        Get the user at a specific rank.

        Uses Rank Tree select(): O(log n)
        """
        tree = self._build_user_rank_tree()
        total = len(tree)

        if rank < 1 or rank > total:
            return None

        # Convert to tree rank (reverse order for "top")
        tree_rank = total - rank + 1
        return tree.select(tree_rank)

    # ==========================================
    # BUCKET SORT (TIME-BASED HISTOGRAMS)
    # ==========================================

    def get_activity_histogram(
        self,
        bucket_size: int = 86400,  # 1 day default
        start_time: int = None,
        end_time: int = None
    ) -> list[tuple[str, int]]:
        """
        Get activity histogram using Bucket Sort.

        O(n + k) where k = number of buckets

        Args:
            bucket_size: Bucket size in seconds (default: 1 day)
            start_time: Start timestamp (default: earliest message)
            end_time: End timestamp (default: latest message)

        Returns:
            List of (date_string, count) tuples
        """
        cursor = self.conn.execute(
            'SELECT id, date_unixtime FROM messages WHERE date_unixtime IS NOT NULL'
        )
        records = [{'id': row[0], 'date_unixtime': row[1]} for row in cursor.fetchall()]

        if not records:
            return []

        hist = time_histogram(records, 'date_unixtime', bucket_size)

        # Format timestamps as dates
        return [
            (datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'), count)
            for ts, count in hist
        ]

    def get_hourly_distribution(self) -> dict[int, int]:
        """
        Get message distribution by hour of day.

        Uses Bucket Sort: O(n)
        """
        cursor = self.conn.execute(
            'SELECT id, date_unixtime FROM messages WHERE date_unixtime IS NOT NULL'
        )
        records = [{'id': row[0], 'date_unixtime': row[1]} for row in cursor.fetchall()]

        return hourly_distribution(records, 'date_unixtime')

    # ==========================================
    # ORIGINAL METHODS (kept for compatibility)
    # ==========================================

    def get_hourly_activity(self) -> dict[int, int]:
        """Get message count by hour of day."""
        sql = '''
            SELECT
                CAST(strftime('%H', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as hour,
                COUNT(*) as count
            FROM messages
            WHERE date_unixtime IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        '''
        cursor = self.conn.execute(sql)
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_daily_activity(self) -> dict[str, int]:
        """Get message count by day of week."""
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        sql = '''
            SELECT
                CAST(strftime('%w', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as day,
                COUNT(*) as count
            FROM messages
            WHERE date_unixtime IS NOT NULL
            GROUP BY day
            ORDER BY day
        '''
        cursor = self.conn.execute(sql)
        return {days[row[0]]: row[1] for row in cursor.fetchall()}

    def get_monthly_activity(self) -> dict[str, int]:
        """Get message count by month."""
        sql = '''
            SELECT
                strftime('%Y-%m', datetime(date_unixtime, 'unixepoch')) as month,
                COUNT(*) as count
            FROM messages
            WHERE date_unixtime IS NOT NULL
            GROUP BY month
            ORDER BY month
        '''
        cursor = self.conn.execute(sql)
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_top_domains(self, limit: int = 20) -> list[tuple[str, int]]:
        """Get most shared domains from links."""
        return self.get_top_domains_heap(limit)

    def get_top_mentioned(self, limit: int = 20) -> list[tuple[str, int]]:
        """Get most mentioned users/channels."""
        sql = '''
            SELECT value, COUNT(*) as count
            FROM entities
            WHERE type = 'mention'
            GROUP BY value
            ORDER BY count DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (limit,))
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_forwarded_sources(self, limit: int = 20) -> list[dict]:
        """Get top sources of forwarded messages."""
        sql = '''
            SELECT
                forwarded_from,
                forwarded_from_id,
                COUNT(*) as count
            FROM messages
            WHERE forwarded_from IS NOT NULL
            GROUP BY forwarded_from_id
            ORDER BY count DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_word_frequency(self, limit: int = 50, min_length: int = 3) -> list[tuple[str, int]]:
        """Get most frequent words using Heap-based Top-K."""
        return self.get_top_words_heap(limit, min_length)

    def get_reply_network(self, limit: int = 100) -> list[dict]:
        """Get reply relationships between users."""
        sql = '''
            SELECT
                m1.from_id as replier_id,
                m1.from_name as replier_name,
                m2.from_id as replied_to_id,
                m2.from_name as replied_to_name,
                COUNT(*) as reply_count
            FROM messages m1
            JOIN messages m2 ON m1.reply_to_message_id = m2.id
            WHERE m1.reply_to_message_id IS NOT NULL
            GROUP BY m1.from_id, m2.from_id
            ORDER BY reply_count DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_user_stats(self, user_id: str) -> dict:
        """Get detailed statistics for a specific user."""
        stats = {}

        cursor = self.conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(has_links) as links,
                SUM(has_media) as media,
                SUM(has_mentions) as mentions,
                SUM(is_edited) as edited,
                MIN(date_unixtime) as first_msg,
                MAX(date_unixtime) as last_msg
            FROM messages WHERE from_id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        stats.update(dict(row))

        cursor = self.conn.execute('''
            SELECT COUNT(*) FROM messages m1
            JOIN messages m2 ON m1.reply_to_message_id = m2.id
            WHERE m2.from_id = ?
        ''', (user_id,))
        stats['replies_received'] = cursor.fetchone()[0]

        cursor = self.conn.execute('''
            SELECT COUNT(*) FROM messages
            WHERE from_id = ? AND reply_to_message_id IS NOT NULL
        ''', (user_id,))
        stats['replies_sent'] = cursor.fetchone()[0]

        # Add rank info using Rank Tree
        rank_info = self.get_user_rank(user_id)
        stats['rank'] = rank_info.get('rank')
        stats['percentile'] = rank_info.get('percentile')

        return stats


def print_bar(value: int, max_value: int, width: int = 40) -> str:
    """Create a simple ASCII bar."""
    if max_value == 0:
        return ''
    bar_length = int((value / max_value) * width)
    return '█' * bar_length + '░' * (width - bar_length)


def main():
    parser = argparse.ArgumentParser(description='Analyze indexed Telegram messages (Enhanced)')
    parser.add_argument('--db', default='telegram.db', help='Database path')

    # Original options
    parser.add_argument('--stats', action='store_true', help='Show general statistics')
    parser.add_argument('--top-users', action='store_true', help='Show top users')
    parser.add_argument('--hourly', action='store_true', help='Show hourly activity')
    parser.add_argument('--daily', action='store_true', help='Show daily activity')
    parser.add_argument('--monthly', action='store_true', help='Show monthly activity')
    parser.add_argument('--domains', action='store_true', help='Show top shared domains')
    parser.add_argument('--mentions', action='store_true', help='Show top mentions')
    parser.add_argument('--words', action='store_true', help='Show word frequency')
    parser.add_argument('--sources', action='store_true', help='Show forwarded message sources')
    parser.add_argument('--replies', action='store_true', help='Show reply network')
    parser.add_argument('--user', help='Show stats for specific user ID')

    # NEW: Enhanced options
    parser.add_argument('--similar', action='store_true', help='Find similar messages (LCS)')
    parser.add_argument('--reposts', action='store_true', help='Find potential reposts')
    parser.add_argument('--percentiles', action='store_true', help='Show message length percentiles')
    parser.add_argument('--response-times', action='store_true', help='Show response time percentiles')
    parser.add_argument('--user-rank', help='Get rank of specific user')
    parser.add_argument('--rank', type=int, help='Get user at specific rank')
    parser.add_argument('--histogram', action='store_true', help='Show activity histogram')
    parser.add_argument('--bucket-size', type=int, default=86400, help='Histogram bucket size in seconds')

    parser.add_argument('--limit', type=int, default=20, help='Limit results')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')

    args = parser.parse_args()

    with TelegramAnalyzer(args.db) as analyzer:
        # === ORIGINAL OPTIONS ===
        if args.stats:
            stats = analyzer.get_stats()
            if args.json:
                print(json.dumps(stats, indent=2, ensure_ascii=False))
            else:
                print("=== General Statistics ===\n")
                print(f"Total messages:      {stats['total_messages']:,}")
                print(f"Total users:         {stats['total_users']:,}")
                print(f"First message:       {stats.get('first_message', 'N/A')}")
                print(f"Last message:        {stats.get('last_message', 'N/A')}")
                print(f"Days span:           {stats.get('days_span', 'N/A')}")
                print(f"Messages with media: {stats['messages_with_media']:,}")
                print(f"Messages with links: {stats['messages_with_links']:,}")
                print(f"Forwarded messages:  {stats['forwarded_messages']:,}")
                print(f"Reply messages:      {stats['reply_messages']:,}")
                if 'median_message_length' in stats:
                    print(f"\nMedian msg length:   {stats['median_message_length']:.0f} chars")
                    print(f"90th percentile:     {stats['p90_message_length']:.0f} chars")
                print(f"\nEntities: {stats.get('entities', {})}")
            return

        if args.top_users:
            users = analyzer.get_top_users(args.limit)
            if args.json:
                print(json.dumps(users, indent=2, ensure_ascii=False))
            else:
                print("=== Top Users by Message Count (Heap-based Top-K) ===\n")
                max_count = users[0]['message_count'] if users else 0
                for i, user in enumerate(users, 1):
                    bar = print_bar(user['message_count'], max_count, 30)
                    print(f"{i:2}. {user['from_name'][:20]:20} {bar} {user['message_count']:,}")
            return

        if args.hourly:
            hourly = analyzer.get_hourly_activity()
            if args.json:
                print(json.dumps(hourly, indent=2))
            else:
                print("=== Hourly Activity ===\n")
                max_count = max(hourly.values()) if hourly else 0
                for hour in range(24):
                    count = hourly.get(hour, 0)
                    bar = print_bar(count, max_count, 40)
                    print(f"{hour:02}:00  {bar} {count:,}")
            return

        if args.daily:
            daily = analyzer.get_daily_activity()
            if args.json:
                print(json.dumps(daily, indent=2))
            else:
                print("=== Daily Activity ===\n")
                max_count = max(daily.values()) if daily else 0
                for day, count in daily.items():
                    bar = print_bar(count, max_count, 40)
                    print(f"{day:10} {bar} {count:,}")
            return

        if args.monthly:
            monthly = analyzer.get_monthly_activity()
            if args.json:
                print(json.dumps(monthly, indent=2))
            else:
                print("=== Monthly Activity ===\n")
                max_count = max(monthly.values()) if monthly else 0
                for month, count in monthly.items():
                    bar = print_bar(count, max_count, 40)
                    print(f"{month}  {bar} {count:,}")
            return

        if args.domains:
            domains = analyzer.get_top_domains(args.limit)
            if args.json:
                print(json.dumps(dict(domains), indent=2))
            else:
                print("=== Top Shared Domains (Heap-based Top-K) ===\n")
                max_count = domains[0][1] if domains else 0
                for domain, count in domains:
                    bar = print_bar(count, max_count, 30)
                    print(f"{domain[:30]:30} {bar} {count:,}")
            return

        if args.mentions:
            mentions = analyzer.get_top_mentioned(args.limit)
            if args.json:
                print(json.dumps(dict(mentions), indent=2))
            else:
                print("=== Top Mentioned Users ===\n")
                max_count = mentions[0][1] if mentions else 0
                for mention, count in mentions:
                    bar = print_bar(count, max_count, 30)
                    print(f"{mention:20} {bar} {count:,}")
            return

        if args.words:
            words = analyzer.get_word_frequency(args.limit)
            if args.json:
                print(json.dumps(dict(words), indent=2, ensure_ascii=False))
            else:
                print("=== Top Words (Heap-based Top-K) ===\n")
                max_count = words[0][1] if words else 0
                for word, count in words:
                    bar = print_bar(count, max_count, 30)
                    print(f"{word:20} {bar} {count:,}")
            return

        if args.sources:
            sources = analyzer.get_forwarded_sources(args.limit)
            if args.json:
                print(json.dumps(sources, indent=2, ensure_ascii=False))
            else:
                print("=== Top Forwarded Sources ===\n")
                max_count = sources[0]['count'] if sources else 0
                for src in sources:
                    bar = print_bar(src['count'], max_count, 30)
                    name = src['forwarded_from'] or 'Unknown'
                    print(f"{name[:30]:30} {bar} {src['count']:,}")
            return

        if args.replies:
            replies = analyzer.get_reply_network(args.limit)
            if args.json:
                print(json.dumps(replies, indent=2, ensure_ascii=False))
            else:
                print("=== Reply Network ===\n")
                for r in replies:
                    print(f"{r['replier_name']} → {r['replied_to_name']}: {r['reply_count']} replies")
            return

        if args.user:
            user_stats = analyzer.get_user_stats(args.user)
            if args.json:
                print(json.dumps(user_stats, indent=2))
            else:
                print(f"=== Stats for {args.user} ===\n")
                for key, value in user_stats.items():
                    print(f"{key}: {value}")
            return

        # === NEW ENHANCED OPTIONS ===

        if args.similar:
            print(f"=== Similar Messages (LCS, threshold={args.threshold}) ===\n")
            similar = analyzer.find_similar_messages(
                threshold=args.threshold,
                limit=args.limit
            )
            if args.json:
                print(json.dumps(similar, indent=2, ensure_ascii=False))
            else:
                for id1, id2, sim, text1, text2 in similar:
                    print(f"Similarity: {sim:.1%}")
                    print(f"  [{id1}] {text1}...")
                    print(f"  [{id2}] {text2}...")
                    print()
            return

        if args.reposts:
            print("=== Potential Reposts (LCS-based) ===\n")
            reposts = analyzer.find_reposts(threshold=args.threshold)
            if args.json:
                print(json.dumps(reposts, indent=2, ensure_ascii=False))
            else:
                for r in reposts[:args.limit]:
                    print(f"Similarity: {r['similarity']:.1%}")
                    print(f"  User 1: {r['user_1']}")
                    print(f"  User 2: {r['user_2']}")
                    print(f"  Text: {r['text_preview']}...")
                    print()
            return

        if args.percentiles:
            print("=== Message Length Percentiles (Selection Algorithm) ===\n")
            stats = analyzer.get_message_length_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                for key, value in stats.items():
                    print(f"{key:15}: {value:,.0f}")
            return

        if args.response_times:
            print("=== Response Time Percentiles (Selection Algorithm) ===\n")
            stats = analyzer.get_response_time_percentiles()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                for key, value in stats.items():
                    if 'seconds' in key:
                        print(f"{key:15}: {value:,.0f}s ({value/60:.1f}m)")
                    else:
                        print(f"{key:15}: {value:,}")
            return

        if args.user_rank:
            print(f"=== User Rank (Rank Tree O(log n)) ===\n")
            rank_info = analyzer.get_user_rank(args.user_rank)
            if args.json:
                print(json.dumps(rank_info, indent=2))
            else:
                print(f"User ID:       {rank_info.get('user_id')}")
                print(f"Message count: {rank_info.get('message_count'):,}")
                print(f"Rank:          #{rank_info.get('rank')} of {rank_info.get('total_users')}")
                print(f"Percentile:    Top {rank_info.get('percentile'):.1f}%")
            return

        if args.rank:
            print(f"=== User at Rank #{args.rank} (Rank Tree O(log n)) ===\n")
            user = analyzer.get_user_by_rank(args.rank)
            if args.json:
                print(json.dumps(user, indent=2, ensure_ascii=False))
            elif user:
                print(f"Name:          {user.get('name')}")
                print(f"User ID:       {user.get('user_id')}")
                print(f"Message count: {user.get('count'):,}")
            else:
                print(f"No user at rank {args.rank}")
            return

        if args.histogram:
            print(f"=== Activity Histogram (Bucket Sort, bucket={args.bucket_size}s) ===\n")
            hist = analyzer.get_activity_histogram(bucket_size=args.bucket_size)
            if args.json:
                print(json.dumps(hist, indent=2))
            else:
                max_count = max(c for _, c in hist) if hist else 0
                for date_str, count in hist[-args.limit:]:
                    bar = print_bar(count, max_count, 40)
                    print(f"{date_str}  {bar} {count:,}")
            return

        # Default: show help
        parser.print_help()


if __name__ == '__main__':
    main()
