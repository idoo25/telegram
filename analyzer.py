#!/usr/bin/env python3
"""
Telegram Chat Analytics
Analysis and statistics for indexed Telegram messages.

Usage:
    python analyzer.py --db telegram.db [options]
    python analyzer.py --stats
    python analyzer.py --top-users
    python analyzer.py --hourly
"""

import sqlite3
import argparse
import json
from collections import Counter
from datetime import datetime
from typing import Optional
import re


class TelegramAnalyzer:
    """Analytics interface for indexed Telegram messages."""

    def __init__(self, db_path: str = 'telegram.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_stats(self) -> dict:
        """Get general statistics about the indexed data."""
        stats = {}

        # Total messages
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages')
        stats['total_messages'] = cursor.fetchone()[0]

        # Total users
        cursor = self.conn.execute('SELECT COUNT(DISTINCT from_id) FROM messages')
        stats['total_users'] = cursor.fetchone()[0]

        # Date range
        cursor = self.conn.execute('''
            SELECT MIN(date_unixtime), MAX(date_unixtime) FROM messages
            WHERE date_unixtime IS NOT NULL
        ''')
        row = cursor.fetchone()
        if row[0] and row[1]:
            stats['first_message'] = datetime.fromtimestamp(row[0]).isoformat()
            stats['last_message'] = datetime.fromtimestamp(row[1]).isoformat()
            stats['days_span'] = (row[1] - row[0]) // 86400

        # Messages with media
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_media = 1')
        stats['messages_with_media'] = cursor.fetchone()[0]

        # Messages with links
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_links = 1')
        stats['messages_with_links'] = cursor.fetchone()[0]

        # Messages with mentions
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE has_mentions = 1')
        stats['messages_with_mentions'] = cursor.fetchone()[0]

        # Forwarded messages
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE forwarded_from IS NOT NULL')
        stats['forwarded_messages'] = cursor.fetchone()[0]

        # Replies
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE reply_to_message_id IS NOT NULL')
        stats['reply_messages'] = cursor.fetchone()[0]

        # Edited messages
        cursor = self.conn.execute('SELECT COUNT(*) FROM messages WHERE is_edited = 1')
        stats['edited_messages'] = cursor.fetchone()[0]

        # Total entities
        cursor = self.conn.execute('SELECT type, COUNT(*) FROM entities GROUP BY type')
        stats['entities'] = {row[0]: row[1] for row in cursor.fetchall()}

        return stats

    def get_top_users(self, limit: int = 20) -> list[dict]:
        """Get most active users by message count."""
        sql = '''
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
            ORDER BY message_count DESC
            LIMIT ?
        '''
        cursor = self.conn.execute(sql, (limit,))
        return [dict(row) for row in cursor.fetchall()]

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
        sql = '''
            SELECT value FROM entities WHERE type = 'link'
        '''
        cursor = self.conn.execute(sql)

        domain_pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
        domains = Counter()

        for row in cursor.fetchall():
            url = row[0]
            match = domain_pattern.match(url)
            if match:
                domains[match.group(1)] += 1

        return domains.most_common(limit)

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
        """
        Get most frequent words in messages.
        Note: Basic tokenization, works for Hebrew and English.
        """
        sql = 'SELECT text_plain FROM messages WHERE text_plain IS NOT NULL'
        cursor = self.conn.execute(sql)

        words = Counter()
        # Simple word pattern that works for Hebrew and English
        word_pattern = re.compile(r'[\u0590-\u05FFa-zA-Z]+')

        for row in cursor.fetchall():
            text = row[0]
            for word in word_pattern.findall(text.lower()):
                if len(word) >= min_length:
                    words[word] += 1

        return words.most_common(limit)

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

        # Basic counts
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

        # Replies received
        cursor = self.conn.execute('''
            SELECT COUNT(*) FROM messages m1
            JOIN messages m2 ON m1.reply_to_message_id = m2.id
            WHERE m2.from_id = ?
        ''', (user_id,))
        stats['replies_received'] = cursor.fetchone()[0]

        # Replies sent
        cursor = self.conn.execute('''
            SELECT COUNT(*) FROM messages
            WHERE from_id = ? AND reply_to_message_id IS NOT NULL
        ''', (user_id,))
        stats['replies_sent'] = cursor.fetchone()[0]

        return stats


def print_bar(value: int, max_value: int, width: int = 40) -> str:
    """Create a simple ASCII bar."""
    if max_value == 0:
        return ''
    bar_length = int((value / max_value) * width)
    return '█' * bar_length + '░' * (width - bar_length)


def main():
    parser = argparse.ArgumentParser(description='Analyze indexed Telegram messages')
    parser.add_argument('--db', default='telegram.db', help='Database path')
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
    parser.add_argument('--limit', type=int, default=20, help='Limit results')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    with TelegramAnalyzer(args.db) as analyzer:
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
                print(f"\nEntities: {stats.get('entities', {})}")
            return

        if args.top_users:
            users = analyzer.get_top_users(args.limit)
            if args.json:
                print(json.dumps(users, indent=2, ensure_ascii=False))
            else:
                print("=== Top Users by Message Count ===\n")
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
                print("=== Top Shared Domains ===\n")
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
                print("=== Top Words ===\n")
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
                    print(f"{src['forwarded_from'][:30]:30} {bar} {src['count']:,}")
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

        # Default: show general stats
        parser.print_help()


if __name__ == '__main__':
    main()
