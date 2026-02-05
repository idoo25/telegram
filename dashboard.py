#!/usr/bin/env python3
"""
Telegram Analytics Dashboard - Web Server

A Flask-based web dashboard for visualizing Telegram chat analytics.
Inspired by Combot and other Telegram statistics bots.

Usage:
    python dashboard.py --db telegram.db --port 5000
    Then open http://localhost:5000 in your browser

Requirements:
    pip install flask
"""

import sqlite3
import json
import csv
import io
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, Response
from typing import Optional
from collections import defaultdict

# ==========================================
# AI CONFIGURATION
# Set via environment variables (e.g. in .env or hosting platform settings)
# ==========================================
if not os.environ.get('AI_PROVIDER'):
    os.environ['AI_PROVIDER'] = 'gemini'
# GEMINI_API_KEY should be set as an environment variable, not hardcoded

# Import our algorithms
from algorithms import (
    TopK, find_median, find_percentile, top_k_frequent,
    RankTree, lcs_similarity, find_similar_messages,
    bucket_sort_by_time, time_histogram, RankedTimeIndex
)

# Import semantic search (uses pre-computed embeddings)
try:
    from semantic_search import get_semantic_search
    HAS_SEMANTIC_SEARCH = True
except ImportError:
    HAS_SEMANTIC_SEARCH = False
    get_semantic_search = None

app = Flask(__name__)
DB_PATH = 'telegram.db'


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_timeframe(timeframe: str) -> tuple[int, int]:
    """Parse timeframe string to Unix timestamps."""
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)

    if timeframe == 'today':
        start = today_start
        end = now
    elif timeframe == 'yesterday':
        start = today_start - timedelta(days=1)
        end = today_start
    elif timeframe == 'week':
        start = today_start - timedelta(days=7)
        end = now
    elif timeframe == 'month':
        start = today_start - timedelta(days=30)
        end = now
    elif timeframe == 'year':
        start = today_start - timedelta(days=365)
        end = now
    elif timeframe == 'all':
        return 0, int(now.timestamp())
    else:
        # Custom range: "start,end" as Unix timestamps
        try:
            parts = timeframe.split(',')
            return int(parts[0]), int(parts[1])
        except:
            return 0, int(now.timestamp())

    return int(start.timestamp()), int(end.timestamp())


# ==========================================
# CACHE INVALIDATION SYSTEM
# ==========================================

_cache_version = 0  # Incremented on DB updates to invalidate all caches


def invalidate_caches():
    """Invalidate all cached data. Call after DB updates (sync, import, etc.)."""
    global _cache_version, _user_rank_tree, _user_rank_tree_timeframe
    _cache_version += 1
    _user_rank_tree = None
    _user_rank_tree_timeframe = None


# ==========================================
# GLOBAL ALGORITHM CACHES
# ==========================================

# RankTree for O(log n) user ranking - rebuilt on demand
_user_rank_tree = None
_user_rank_tree_timeframe = None
_user_rank_tree_version = -1

def get_user_rank_tree(timeframe: str):
    """
    Get or rebuild the user rank tree for efficient O(log n) rank queries.
    Tree is cached and rebuilt only when timeframe or DB version changes.
    """
    global _user_rank_tree, _user_rank_tree_timeframe, _user_rank_tree_version

    if (_user_rank_tree is not None
            and _user_rank_tree_timeframe == timeframe
            and _user_rank_tree_version == _cache_version):
        return _user_rank_tree

    start_ts, end_ts = parse_timeframe(timeframe)
    conn = get_db()

    cursor = conn.execute('''
        SELECT from_id, from_name, COUNT(*) as message_count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND from_id IS NOT NULL AND from_id != ''
        GROUP BY from_id
        ORDER BY message_count DESC
    ''', (start_ts, end_ts))

    _user_rank_tree = RankTree()
    for row in cursor.fetchall():
        _user_rank_tree.insert(
            -row['message_count'],
            {'user_id': row['from_id'], 'name': row['from_name'], 'messages': row['message_count']}
        )

    conn.close()
    _user_rank_tree_timeframe = timeframe
    _user_rank_tree_version = _cache_version
    return _user_rank_tree


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/users')
def users_page():
    """User leaderboard page."""
    return render_template('users.html')


@app.route('/moderation')
def moderation_page():
    """Moderation analytics page."""
    return render_template('moderation.html')


@app.route('/search')
def search_page():
    """Search page."""
    return render_template('search.html')


@app.route('/chat')
def chat_page():
    """Chat view page - Telegram-like interface."""
    return render_template('chat.html')


@app.route('/user/<user_id>')
def user_profile_page(user_id):
    """User profile page with comprehensive statistics."""
    return render_template('user_profile.html', user_id=user_id)


@app.route('/settings')
def settings_page():
    """Settings and data update page."""
    return render_template('settings.html')


# ==========================================
# API ENDPOINTS - OVERVIEW STATS
# ==========================================

@app.route('/api/overview')
def api_overview():
    """Get overview statistics."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    # Total messages
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))
    total_messages = cursor.fetchone()[0]

    # Active users
    cursor = conn.execute('''
        SELECT COUNT(DISTINCT from_id) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))
    active_users = cursor.fetchone()[0]

    # Total users (all time)
    cursor = conn.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]

    # Date range
    cursor = conn.execute('''
        SELECT MIN(date_unixtime), MAX(date_unixtime) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))
    row = cursor.fetchone()
    first_msg = row[0] or start_ts
    last_msg = row[1] or end_ts

    # Calculate days
    days = max(1, (last_msg - first_msg) // 86400)

    # Messages per day
    messages_per_day = total_messages / days

    # Users per day (average unique users)
    cursor = conn.execute('''
        SELECT COUNT(DISTINCT from_id) as users,
               date(datetime(date_unixtime, 'unixepoch')) as day
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY day
    ''', (start_ts, end_ts))
    daily_users = [r[0] for r in cursor.fetchall()]
    users_per_day = sum(daily_users) / len(daily_users) if daily_users else 0

    # Messages with media/links
    cursor = conn.execute('''
        SELECT
            SUM(has_media) as media,
            SUM(has_links) as links,
            SUM(has_mentions) as mentions
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))
    row = cursor.fetchone()
    media_count = row[0] or 0
    links_count = row[1] or 0
    mentions_count = row[2] or 0

    # Replies
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND reply_to_message_id IS NOT NULL
    ''', (start_ts, end_ts))
    replies_count = cursor.fetchone()[0]

    # Forwards
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND forwarded_from IS NOT NULL
    ''', (start_ts, end_ts))
    forwards_count = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        'total_messages': total_messages,
        'active_users': active_users,
        'total_users': total_users,
        'messages_per_day': round(messages_per_day, 1),
        'users_per_day': round(users_per_day, 1),
        'messages_per_user': round(total_messages / active_users, 1) if active_users else 0,
        'media_count': media_count,
        'links_count': links_count,
        'mentions_count': mentions_count,
        'replies_count': replies_count,
        'forwards_count': forwards_count,
        'days_span': days,
        'first_message': first_msg,
        'last_message': last_msg
    })


# ==========================================
# API ENDPOINTS - CHARTS
# ==========================================

@app.route('/api/chart/messages')
def api_chart_messages():
    """Get message volume over time."""
    timeframe = request.args.get('timeframe', 'month')
    granularity = request.args.get('granularity', 'day')  # hour, day, week
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    if granularity == 'hour':
        format_str = '%Y-%m-%d %H:00'
    elif granularity == 'week':
        format_str = '%Y-W%W'
    else:  # day
        format_str = '%Y-%m-%d'

    cursor = conn.execute(f'''
        SELECT
            strftime('{format_str}', datetime(date_unixtime, 'unixepoch')) as period,
            COUNT(*) as count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY period
        ORDER BY period
    ''', (start_ts, end_ts))

    data = [{'label': row[0], 'value': row[1]} for row in cursor.fetchall()]
    conn.close()

    return jsonify(data)


@app.route('/api/chart/users')
def api_chart_users():
    """Get active users over time."""
    timeframe = request.args.get('timeframe', 'month')
    granularity = request.args.get('granularity', 'day')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    if granularity == 'hour':
        format_str = '%Y-%m-%d %H:00'
    elif granularity == 'week':
        format_str = '%Y-W%W'
    else:
        format_str = '%Y-%m-%d'

    cursor = conn.execute(f'''
        SELECT
            strftime('{format_str}', datetime(date_unixtime, 'unixepoch')) as period,
            COUNT(DISTINCT from_id) as count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY period
        ORDER BY period
    ''', (start_ts, end_ts))

    data = [{'label': row[0], 'value': row[1]} for row in cursor.fetchall()]
    conn.close()

    return jsonify(data)


@app.route('/api/chart/heatmap')
def api_chart_heatmap():
    """Get activity heatmap (hour of day vs day of week)."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT
            CAST(strftime('%w', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as dow,
            CAST(strftime('%H', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as hour,
            COUNT(*) as count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY dow, hour
    ''', (start_ts, end_ts))

    # Initialize grid
    heatmap = [[0 for _ in range(24)] for _ in range(7)]

    for row in cursor.fetchall():
        dow, hour, count = row
        heatmap[dow][hour] = count

    conn.close()

    return jsonify({
        'data': heatmap,
        'days': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
        'hours': list(range(24))
    })


@app.route('/api/chart/daily')
def api_chart_daily():
    """Get activity by day of week."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    cursor = conn.execute('''
        SELECT
            CAST(strftime('%w', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as dow,
            COUNT(*) as count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY dow
        ORDER BY dow
    ''', (start_ts, end_ts))

    data = {days[row[0]]: row[1] for row in cursor.fetchall()}
    conn.close()

    return jsonify([{'label': day, 'value': data.get(day, 0)} for day in days])


@app.route('/api/chart/hourly')
def api_chart_hourly():
    """Get activity by hour of day."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT
            CAST(strftime('%H', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as hour,
            COUNT(*) as count
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        GROUP BY hour
        ORDER BY hour
    ''', (start_ts, end_ts))

    data = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    return jsonify([{'label': f'{h:02d}:00', 'value': data.get(h, 0)} for h in range(24)])


# ==========================================
# API ENDPOINTS - USERS
# ==========================================

@app.route('/api/users')
def api_users():
    """Get user leaderboard including participants who never sent messages."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    include_inactive = request.args.get('include_inactive', '1') == '1'

    conn = get_db()

    # Get total messages for percentage calculation
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))
    total_messages = cursor.fetchone()[0]

    # Get user stats from messages
    cursor = conn.execute('''
        SELECT
            from_id,
            from_name,
            COUNT(*) as message_count,
            SUM(LENGTH(text_plain)) as char_count,
            SUM(has_links) as links,
            SUM(has_media) as media,
            MIN(date_unixtime) as first_seen,
            MAX(date_unixtime) as last_seen,
            COUNT(DISTINCT date(datetime(date_unixtime, 'unixepoch'))) as active_days
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND from_id IS NOT NULL AND from_id != ''
        GROUP BY from_id
        ORDER BY message_count DESC
    ''', (start_ts, end_ts))

    active_users = []
    active_user_ids = set()
    for row in cursor.fetchall():
        active_user_ids.add(row['from_id'])
        active_users.append({
            'user_id': row['from_id'],
            'name': row['from_name'] or 'Unknown',
            'messages': row['message_count'],
            'characters': row['char_count'] or 0,
            'percentage': round(100 * row['message_count'] / total_messages, 2) if total_messages else 0,
            'links': row['links'] or 0,
            'media': row['media'] or 0,
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'active_days': row['active_days'],
            'daily_average': round(row['message_count'] / max(1, row['active_days']), 1),
            'is_participant': False,
            'role': None,
        })

    # Try to enrich with participant data and add inactive participants
    has_participants = False
    try:
        cursor = conn.execute('SELECT COUNT(*) FROM participants')
        has_participants = cursor.fetchone()[0] > 0
    except Exception:
        pass

    if has_participants:
        # Enrich active users with participant data
        participant_map = {}
        cursor = conn.execute('SELECT * FROM participants')
        for row in cursor.fetchall():
            participant_map[row['user_id']] = dict(row)

        for user in active_users:
            p = participant_map.get(user['user_id'])
            if p:
                user['is_participant'] = True
                user['username'] = p.get('username', '')
                if p.get('is_creator'):
                    user['role'] = 'creator'
                elif p.get('is_admin'):
                    user['role'] = 'admin'
                elif p.get('is_bot'):
                    user['role'] = 'bot'

        # Add inactive participants (those who never sent messages)
        if include_inactive:
            for uid, p in participant_map.items():
                if uid not in active_user_ids:
                    name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                    role = None
                    if p.get('is_creator'):
                        role = 'creator'
                    elif p.get('is_admin'):
                        role = 'admin'
                    elif p.get('is_bot'):
                        role = 'bot'

                    active_users.append({
                        'user_id': uid,
                        'name': name or 'Unknown',
                        'messages': 0,
                        'characters': 0,
                        'percentage': 0,
                        'links': 0,
                        'media': 0,
                        'first_seen': None,
                        'last_seen': None,
                        'active_days': 0,
                        'daily_average': 0,
                        'is_participant': True,
                        'username': p.get('username', ''),
                        'role': role,
                    })

    # Assign ranks (active users first, then inactive)
    users_with_rank = []
    for i, user in enumerate(active_users):
        user['rank'] = i + 1 if user['messages'] > 0 else None
        users_with_rank.append(user)

    total_users = len(users_with_rank)
    total_active = len(active_user_ids)

    # Apply pagination
    page_users = users_with_rank[offset:offset + limit]

    conn.close()

    return jsonify({
        'users': page_users,
        'total': total_users,
        'total_active': total_active,
        'total_participants': total_users - total_active if has_participants else 0,
        'limit': limit,
        'offset': offset
    })


@app.route('/api/user/<user_id>')
def api_user_detail(user_id):
    """Get detailed stats for a specific user."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    # Basic stats
    cursor = conn.execute('''
        SELECT
            from_name,
            COUNT(*) as messages,
            SUM(LENGTH(text_plain)) as characters,
            SUM(has_links) as links,
            SUM(has_media) as media,
            SUM(has_mentions) as mentions,
            MIN(date_unixtime) as first_seen,
            MAX(date_unixtime) as last_seen,
            COUNT(DISTINCT date(datetime(date_unixtime, 'unixepoch'))) as active_days
        FROM messages
        WHERE from_id = ?
        AND date_unixtime BETWEEN ? AND ?
    ''', (user_id, start_ts, end_ts))
    row = cursor.fetchone()

    if not row or not row['messages']:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    # Replies sent
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE from_id = ? AND reply_to_message_id IS NOT NULL
        AND date_unixtime BETWEEN ? AND ?
    ''', (user_id, start_ts, end_ts))
    replies_sent = cursor.fetchone()[0]

    # Replies received
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages m1
        JOIN messages m2 ON m1.reply_to_message_id = m2.id
        WHERE m2.from_id = ?
        AND m1.date_unixtime BETWEEN ? AND ?
    ''', (user_id, start_ts, end_ts))
    replies_received = cursor.fetchone()[0]

    # Activity by hour
    cursor = conn.execute('''
        SELECT
            CAST(strftime('%H', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as hour,
            COUNT(*) as count
        FROM messages
        WHERE from_id = ?
        AND date_unixtime BETWEEN ? AND ?
        GROUP BY hour
    ''', (user_id, start_ts, end_ts))
    hourly = {row[0]: row[1] for row in cursor.fetchall()}

    # Activity over time
    cursor = conn.execute('''
        SELECT
            date(datetime(date_unixtime, 'unixepoch')) as day,
            COUNT(*) as count
        FROM messages
        WHERE from_id = ?
        AND date_unixtime BETWEEN ? AND ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
    ''', (user_id, start_ts, end_ts))
    daily = [{'date': r[0], 'count': r[1]} for r in cursor.fetchall()]

    # Rank
    cursor = conn.execute('''
        SELECT COUNT(*) + 1 FROM (
            SELECT from_id, COUNT(*) as cnt FROM messages
            WHERE date_unixtime BETWEEN ? AND ?
            GROUP BY from_id
        ) WHERE cnt > ?
    ''', (start_ts, end_ts, row['messages']))
    rank = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        'user_id': user_id,
        'name': row['from_name'] or 'Unknown',
        'messages': row['messages'],
        'characters': row['characters'] or 0,
        'links': row['links'] or 0,
        'media': row['media'] or 0,
        'mentions': row['mentions'] or 0,
        'first_seen': row['first_seen'],
        'last_seen': row['last_seen'],
        'active_days': row['active_days'],
        'daily_average': round(row['messages'] / max(1, row['active_days']), 1),
        'replies_sent': replies_sent,
        'replies_received': replies_received,
        'rank': rank,
        'hourly_activity': [hourly.get(h, 0) for h in range(24)],
        'daily_activity': daily
    })


@app.route('/api/user/<user_id>/profile')
def api_user_profile(user_id):
    """Get comprehensive user profile with all available statistics."""
    conn = get_db()

    # ---- Participant info (from Telethon sync) ----
    participant = None
    try:
        cursor = conn.execute('SELECT * FROM participants WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        if row:
            participant = dict(row)
    except Exception:
        pass  # Table might not exist yet

    # ---- Basic message stats ----
    cursor = conn.execute('''
        SELECT
            from_name,
            COUNT(*) as total_messages,
            SUM(text_length) as total_chars,
            AVG(text_length) as avg_length,
            MAX(text_length) as max_length,
            SUM(has_links) as links_shared,
            SUM(has_media) as media_sent,
            SUM(has_photo) as photos_sent,
            SUM(has_mentions) as mentions_made,
            SUM(is_edited) as edits,
            MIN(date_unixtime) as first_message,
            MAX(date_unixtime) as last_message,
            COUNT(DISTINCT date(datetime(date_unixtime, 'unixepoch'))) as active_days
        FROM messages WHERE from_id = ?
    ''', (user_id,))
    stats = cursor.fetchone()

    if not stats or not stats['total_messages']:
        # User might be a participant who never sent a message
        if participant:
            conn.close()
            return jsonify({
                'user_id': user_id,
                'participant': participant,
                'has_messages': False,
                'name': f"{participant.get('first_name', '')} {participant.get('last_name', '')}".strip()
            })
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    stats = dict(stats)

    # ---- Replies sent (who does this user reply to most) ----
    cursor = conn.execute('''
        SELECT r.from_name, r.from_id, COUNT(*) as cnt
        FROM messages m
        JOIN messages r ON m.reply_to_message_id = r.id
        WHERE m.from_id = ? AND r.from_id != ?
        GROUP BY r.from_id
        ORDER BY cnt DESC
        LIMIT 10
    ''', (user_id, user_id))
    replies_to = [{'name': r[0], 'user_id': r[1], 'count': r[2]} for r in cursor.fetchall()]

    # ---- Replies received (who replies to this user most) ----
    cursor = conn.execute('''
        SELECT m.from_name, m.from_id, COUNT(*) as cnt
        FROM messages m
        JOIN messages r ON m.reply_to_message_id = r.id
        WHERE r.from_id = ? AND m.from_id != ?
        GROUP BY m.from_id
        ORDER BY cnt DESC
        LIMIT 10
    ''', (user_id, user_id))
    replies_from = [{'name': r[0], 'user_id': r[1], 'count': r[2]} for r in cursor.fetchall()]

    # ---- Total replies sent/received ----
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE from_id = ? AND reply_to_message_id IS NOT NULL
    ''', (user_id,))
    total_replies_sent = cursor.fetchone()[0]

    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages m
        JOIN messages r ON m.reply_to_message_id = r.id
        WHERE r.from_id = ? AND m.from_id != ?
    ''', (user_id, user_id))
    total_replies_received = cursor.fetchone()[0]

    # ---- Forwarded messages ----
    cursor = conn.execute('''
        SELECT COUNT(*) FROM messages
        WHERE from_id = ? AND forwarded_from IS NOT NULL
    ''', (user_id,))
    forwards_sent = cursor.fetchone()[0]

    # ---- Top forwarded sources ----
    cursor = conn.execute('''
        SELECT forwarded_from, COUNT(*) as cnt
        FROM messages
        WHERE from_id = ? AND forwarded_from IS NOT NULL
        GROUP BY forwarded_from
        ORDER BY cnt DESC
        LIMIT 5
    ''', (user_id,))
    top_forward_sources = [{'name': r[0], 'count': r[1]} for r in cursor.fetchall()]

    # ---- Activity by hour ----
    cursor = conn.execute('''
        SELECT
            CAST(strftime('%H', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as hour,
            COUNT(*) as count
        FROM messages WHERE from_id = ?
        GROUP BY hour
    ''', (user_id,))
    hourly = {r[0]: r[1] for r in cursor.fetchall()}

    # ---- Activity by weekday ----
    cursor = conn.execute('''
        SELECT
            CAST(strftime('%w', datetime(date_unixtime, 'unixepoch')) AS INTEGER) as weekday,
            COUNT(*) as count
        FROM messages WHERE from_id = ?
        GROUP BY weekday
    ''', (user_id,))
    weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekday_data = {r[0]: r[1] for r in cursor.fetchall()}
    weekday_activity = [{'day': weekday_names[d], 'count': weekday_data.get(d, 0)} for d in range(7)]

    # ---- Activity trend (last 90 days) ----
    cursor = conn.execute('''
        SELECT
            date(datetime(date_unixtime, 'unixepoch')) as day,
            COUNT(*) as count
        FROM messages WHERE from_id = ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 90
    ''', (user_id,))
    daily_activity = [{'date': r[0], 'count': r[1]} for r in cursor.fetchall()]

    # ---- Monthly trend ----
    cursor = conn.execute('''
        SELECT
            strftime('%Y-%m', datetime(date_unixtime, 'unixepoch')) as month,
            COUNT(*) as count
        FROM messages WHERE from_id = ?
        GROUP BY month
        ORDER BY month
    ''', (user_id,))
    monthly_activity = [{'month': r[0], 'count': r[1]} for r in cursor.fetchall()]

    # ---- Top links shared ----
    cursor = conn.execute('''
        SELECT e.value, COUNT(*) as cnt
        FROM entities e
        JOIN messages m ON e.message_id = m.id
        WHERE m.from_id = ? AND e.type = 'link'
        GROUP BY e.value
        ORDER BY cnt DESC
        LIMIT 10
    ''', (user_id,))
    top_links = [{'url': r[0], 'count': r[1]} for r in cursor.fetchall()]

    # ---- Rank among all users ----
    cursor = conn.execute('''
        SELECT COUNT(*) + 1 FROM (
            SELECT from_id, COUNT(*) as cnt FROM messages GROUP BY from_id
        ) WHERE cnt > ?
    ''', (stats['total_messages'],))
    rank = cursor.fetchone()[0]

    cursor = conn.execute('SELECT COUNT(DISTINCT from_id) FROM messages')
    total_users = cursor.fetchone()[0]

    # ---- Average reply time (when replying to someone) ----
    cursor = conn.execute('''
        SELECT AVG(m.date_unixtime - r.date_unixtime)
        FROM messages m
        JOIN messages r ON m.reply_to_message_id = r.id
        WHERE m.from_id = ?
        AND m.date_unixtime - r.date_unixtime > 0
        AND m.date_unixtime - r.date_unixtime < 86400
    ''', (user_id,))
    avg_reply_time = cursor.fetchone()[0]

    conn.close()

    # ---- Build response ----
    total_msgs = stats['total_messages']
    active_days = stats['active_days'] or 1
    first_msg = stats['first_message']
    last_msg = stats['last_message']
    span_days = max(1, (last_msg - first_msg) / 86400) if first_msg and last_msg else 1

    return jsonify({
        'user_id': user_id,
        'name': stats['from_name'] or 'Unknown',
        'has_messages': True,
        'participant': participant,

        # Core stats
        'total_messages': total_msgs,
        'total_characters': stats['total_chars'] or 0,
        'avg_message_length': round(stats['avg_length'] or 0, 1),
        'max_message_length': stats['max_length'] or 0,
        'links_shared': stats['links_shared'] or 0,
        'media_sent': stats['media_sent'] or 0,
        'photos_sent': stats['photos_sent'] or 0,
        'mentions_made': stats['mentions_made'] or 0,
        'edits': stats['edits'] or 0,
        'forwards_sent': forwards_sent,

        # Time stats
        'first_message': first_msg,
        'last_message': last_msg,
        'active_days': active_days,
        'daily_average': round(total_msgs / active_days, 1),
        'messages_per_calendar_day': round(total_msgs / span_days, 1),

        # Reply stats
        'total_replies_sent': total_replies_sent,
        'total_replies_received': total_replies_received,
        'reply_ratio': round(total_replies_sent / max(1, total_msgs) * 100, 1),
        'avg_reply_time_seconds': round(avg_reply_time) if avg_reply_time else None,
        'replies_to': replies_to,
        'replies_from': replies_from,

        # Forward stats
        'top_forward_sources': top_forward_sources,

        # Ranking
        'rank': rank,
        'total_active_users': total_users,

        # Activity patterns
        'hourly_activity': [hourly.get(h, 0) for h in range(24)],
        'weekday_activity': weekday_activity,
        'daily_activity': daily_activity,
        'monthly_activity': monthly_activity,

        # Content
        'top_links': top_links,
    })


# ==========================================
# API ENDPOINTS - CONTENT ANALYTICS
# ==========================================

@app.route('/api/top/words')
def api_top_words():
    """Get top words."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 30))

    conn = get_db()

    cursor = conn.execute('''
        SELECT text_plain FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND text_plain IS NOT NULL
    ''', (start_ts, end_ts))

    import re
    word_pattern = re.compile(r'[\u0590-\u05FFa-zA-Z]{3,}')
    words = []

    for row in cursor.fetchall():
        words.extend(word_pattern.findall(row[0].lower()))

    conn.close()

    top_words = top_k_frequent(words, limit)
    return jsonify([{'word': w, 'count': c} for w, c in top_words])


@app.route('/api/top/domains')
def api_top_domains():
    """Get top shared domains."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 20))

    conn = get_db()

    cursor = conn.execute('''
        SELECT e.value FROM entities e
        JOIN messages m ON e.message_id = m.id
        WHERE e.type = 'link'
        AND m.date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))

    import re
    domain_pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
    domains = []

    for row in cursor.fetchall():
        match = domain_pattern.match(row[0])
        if match:
            domains.append(match.group(1))

    conn.close()

    top_domains = top_k_frequent(domains, limit)
    return jsonify([{'domain': d, 'count': c} for d, c in top_domains])


@app.route('/api/top/mentions')
def api_top_mentions():
    """Get top mentioned users."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 20))

    conn = get_db()

    cursor = conn.execute('''
        SELECT e.value, COUNT(*) as count FROM entities e
        JOIN messages m ON e.message_id = m.id
        WHERE e.type = 'mention'
        AND m.date_unixtime BETWEEN ? AND ?
        GROUP BY e.value
        ORDER BY count DESC
        LIMIT ?
    ''', (start_ts, end_ts, limit))

    data = [{'mention': row[0], 'count': row[1]} for row in cursor.fetchall()]
    conn.close()

    return jsonify(data)


# ==========================================
# API ENDPOINTS - ADVANCED ANALYTICS (Course Algorithms)
# ==========================================

@app.route('/api/similar/<int:message_id>')
def api_similar_messages(message_id):
    """
    Find messages similar to a given message using LCS algorithm.

    Algorithm: LCS (Longest Common Subsequence)
    Time: O(n * m) where n = sample size, m = avg message length
    Use case: Detect reposts, spam, similar content
    """
    threshold = float(request.args.get('threshold', 0.7))
    limit = int(request.args.get('limit', 10))
    sample_size = int(request.args.get('sample', 1000))

    conn = get_db()

    # Get the target message
    cursor = conn.execute('''
        SELECT text_plain, from_name, date FROM messages WHERE id = ?
    ''', (message_id,))
    target = cursor.fetchone()

    if not target or not target['text_plain']:
        conn.close()
        return jsonify({'error': 'Message not found or empty'}), 404

    target_text = target['text_plain']

    # Get sample of messages to compare (excluding the target)
    cursor = conn.execute('''
        SELECT id, text_plain, from_name, date FROM messages
        WHERE id != ? AND text_plain IS NOT NULL AND LENGTH(text_plain) > 20
        ORDER BY RANDOM()
        LIMIT ?
    ''', (message_id, sample_size))

    messages = [(row['id'], row['text_plain']) for row in cursor.fetchall()]
    conn.close()

    # Find similar messages using LCS
    similar = []
    for msg_id, text in messages:
        sim = lcs_similarity(target_text, text)
        if sim >= threshold:
            similar.append({
                'id': msg_id,
                'similarity': round(sim * 100, 1),
                'text': text[:200] + '...' if len(text) > 200 else text
            })

    # Sort by similarity descending and limit
    similar.sort(key=lambda x: x['similarity'], reverse=True)
    similar = similar[:limit]

    return jsonify({
        'target': {
            'id': message_id,
            'text': target_text[:200] + '...' if len(target_text) > 200 else target_text,
            'from': target['from_name'],
            'date': target['date']
        },
        'similar': similar,
        'algorithm': 'LCS (Longest Common Subsequence)',
        'threshold': threshold
    })


@app.route('/api/analytics/similar')
def api_find_all_similar():
    """
    Find all similar message pairs in the database.

    Algorithm: LCS with early termination
    Time: O(n² * m) where n = sample size, m = avg message length
    Use case: Detect spam campaigns, repeated content
    """
    timeframe = request.args.get('timeframe', 'all')
    threshold = float(request.args.get('threshold', 0.8))
    sample_size = int(request.args.get('sample', 500))
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT id, text_plain, from_name, from_id FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND text_plain IS NOT NULL AND LENGTH(text_plain) > 30
        ORDER BY RANDOM()
        LIMIT ?
    ''', (start_ts, end_ts, sample_size))

    messages = [(row['id'], row['text_plain'], row['from_name'], row['from_id'])
                for row in cursor.fetchall()]
    conn.close()

    # Use our LCS algorithm to find similar pairs
    message_pairs = [(id_, text) for id_, text, _, _ in messages]
    similar_pairs = find_similar_messages(message_pairs, threshold=threshold, min_length=30)

    # Build result with user info
    id_to_info = {id_: (name, uid) for id_, _, name, uid in messages}
    id_to_text = {id_: text for id_, text, _, _ in messages}

    results = []
    for id1, id2, sim in similar_pairs[:50]:  # Limit to top 50
        results.append({
            'message1': {
                'id': id1,
                'text': id_to_text[id1][:150],
                'from': id_to_info[id1][0]
            },
            'message2': {
                'id': id2,
                'text': id_to_text[id2][:150],
                'from': id_to_info[id2][0]
            },
            'similarity': round(sim * 100, 1)
        })

    return jsonify({
        'pairs': results,
        'total_found': len(similar_pairs),
        'algorithm': 'LCS (Longest Common Subsequence)',
        'threshold': threshold,
        'sample_size': sample_size
    })


@app.route('/api/user/rank/<user_id>')
def api_user_rank_efficient(user_id):
    """
    Get user rank using RankTree for O(log n) lookup.

    Algorithm: Order Statistics Tree (AVL-based Rank Tree)
    Time: O(log n) instead of O(n) SQL scan
    Use case: Real-time user ranking queries
    """
    timeframe = request.args.get('timeframe', 'all')
    tree = get_user_rank_tree(timeframe)

    # Find user in tree by iterating (still O(n) for lookup, but rank is O(log n))
    # For true O(log n), we'd need to store user_id as key
    start_ts, end_ts = parse_timeframe(timeframe)
    conn = get_db()

    cursor = conn.execute('''
        SELECT COUNT(*) as count FROM messages
        WHERE from_id = ? AND date_unixtime BETWEEN ? AND ?
    ''', (user_id, start_ts, end_ts))
    user_count = cursor.fetchone()['count']

    if user_count == 0:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    # Use rank tree to find rank (O(log n))
    rank = tree.rank(-user_count)  # Negative because tree uses negative counts

    # Get total users
    total = len(tree)

    conn.close()

    return jsonify({
        'user_id': user_id,
        'messages': user_count,
        'rank': rank,
        'total_users': total,
        'percentile': round(100 * (total - rank + 1) / total, 1) if total > 0 else 0,
        'algorithm': 'RankTree (Order Statistics Tree)',
        'complexity': 'O(log n)'
    })


@app.route('/api/user/by-rank/<int:rank>')
def api_user_by_rank(rank):
    """
    Get user at specific rank using RankTree.

    Algorithm: Order Statistics Tree select(k)
    Time: O(log n)
    Use case: "Who is the 10th most active user?"
    """
    timeframe = request.args.get('timeframe', 'all')
    tree = get_user_rank_tree(timeframe)

    if rank < 1 or rank > len(tree):
        return jsonify({'error': f'Rank must be between 1 and {len(tree)}'}), 400

    user = tree.select(rank)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'rank': rank,
        'user': user,
        'total_users': len(tree),
        'algorithm': 'RankTree select(k)',
        'complexity': 'O(log n)'
    })


@app.route('/api/analytics/histogram')
def api_activity_histogram():
    """
    Get activity histogram using Bucket Sort.

    Algorithm: Bucket Sort
    Time: O(n + k) where k = number of buckets
    Use case: Efficient time-based grouping without SQL GROUP BY
    """
    timeframe = request.args.get('timeframe', 'month')
    bucket_seconds = int(request.args.get('bucket', 86400))  # Default: 1 day
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT date_unixtime FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
    ''', (start_ts, end_ts))

    records = [{'date_unixtime': row[0]} for row in cursor.fetchall()]
    conn.close()

    # Use bucket sort algorithm
    histogram = time_histogram(records, 'date_unixtime', bucket_size=bucket_seconds)

    # Format for frontend
    from datetime import datetime
    result = []
    for bucket_time, count in histogram:
        result.append({
            'timestamp': bucket_time,
            'date': datetime.fromtimestamp(bucket_time).strftime('%Y-%m-%d %H:%M'),
            'count': count
        })

    return jsonify({
        'histogram': result,
        'bucket_size_seconds': bucket_seconds,
        'total_records': len(records),
        'algorithm': 'Bucket Sort',
        'complexity': 'O(n + k)'
    })


@app.route('/api/analytics/percentiles')
def api_message_percentiles():
    """
    Get message length percentiles using Selection Algorithm.

    Algorithm: Quickselect with Median of Medians
    Time: O(n) guaranteed
    Use case: Analyze message length distribution without sorting
    """
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT LENGTH(text_plain) as length FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND text_plain IS NOT NULL
    ''', (start_ts, end_ts))

    lengths = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()

    if not lengths:
        return jsonify({'error': 'No messages found'}), 404

    # Use our O(n) selection algorithm
    result = {
        'count': len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'median': find_median(lengths),
        'p25': find_percentile(lengths, 25),
        'p75': find_percentile(lengths, 75),
        'p90': find_percentile(lengths, 90),
        'p95': find_percentile(lengths, 95),
        'p99': find_percentile(lengths, 99),
        'algorithm': 'Quickselect with Median of Medians',
        'complexity': 'O(n) guaranteed'
    }

    return jsonify(result)


# ==========================================
# API ENDPOINTS - SEARCH
# ==========================================

@app.route('/api/search')
def api_search():
    """Search messages."""
    query = request.args.get('q', '')
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    if not query:
        return jsonify({'results': [], 'total': 0})

    conn = get_db()

    cursor = conn.execute('''
        SELECT
            m.id,
            m.date,
            m.from_name,
            m.from_id,
            m.text_plain,
            m.has_links,
            m.has_media
        FROM messages_fts
        JOIN messages m ON messages_fts.rowid = m.id
        WHERE messages_fts MATCH ?
        AND m.date_unixtime BETWEEN ? AND ?
        ORDER BY m.date_unixtime DESC
        LIMIT ? OFFSET ?
    ''', (query, start_ts, end_ts, limit, offset))

    results = [{
        'id': row['id'],
        'date': row['date'],
        'from_name': row['from_name'],
        'from_id': row['from_id'],
        'text': row['text_plain'][:300] if row['text_plain'] else '',
        'has_links': bool(row['has_links']),
        'has_media': bool(row['has_media'])
    } for row in cursor.fetchall()]

    conn.close()

    return jsonify({
        'results': results,
        'query': query,
        'limit': limit,
        'offset': offset
    })


# ==========================================
# API ENDPOINTS - CHAT VIEW
# ==========================================

@app.route('/api/chat/messages')
def api_chat_messages():
    """Get messages for chat view with filters."""
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 50))
    user_id = request.args.get('user_id')
    search = request.args.get('search')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    has_media = request.args.get('has_media')
    has_link = request.args.get('has_link')

    conn = get_db()

    # Build query
    conditions = ["1=1"]
    params = []

    if user_id:
        conditions.append("m.from_id = ?")
        params.append(user_id)

    if date_from:
        conditions.append("m.date >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("m.date <= ?")
        params.append(date_to)

    if has_media == '1':
        conditions.append("m.has_media = 1")
    elif has_media == '0':
        conditions.append("m.has_media = 0")

    if has_link == '1':
        conditions.append("m.has_links = 1")

    # Handle FTS search
    if search:
        conditions.append("""m.id IN (
            SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?
        )""")
        params.append(search)

    where_clause = " AND ".join(conditions)

    # Get total count
    cursor = conn.execute(f"SELECT COUNT(*) FROM messages m WHERE {where_clause}", params)
    total = cursor.fetchone()[0]

    # Get messages with reply info
    query = f"""
        SELECT
            m.id,
            m.id as message_id,
            m.date,
            m.from_id,
            m.from_name,
            m.text_plain as text,
            m.reply_to_message_id,
            m.forwarded_from,
            m.forwarded_from_id,
            m.has_media,
            m.has_photo,
            m.has_links as has_link,
            m.has_mentions,
            m.is_edited,
            r.from_name as reply_to_name,
            substr(r.text_plain, 1, 100) as reply_to_text
        FROM messages m
        LEFT JOIN messages r ON m.reply_to_message_id = r.id
        WHERE {where_clause}
        ORDER BY m.date ASC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor = conn.execute(query, params)
    messages = [dict(row) for row in cursor.fetchall()]

    # Fetch entities (links, mentions) for these messages
    if messages:
        msg_ids = [m['id'] for m in messages]
        placeholders = ','.join('?' * len(msg_ids))
        ent_cursor = conn.execute(f"""
            SELECT message_id, type, value
            FROM entities
            WHERE message_id IN ({placeholders})
        """, msg_ids)

        # Group entities by message_id
        entities_map = {}
        for row in ent_cursor.fetchall():
            mid = row[0]
            if mid not in entities_map:
                entities_map[mid] = []
            entities_map[mid].append({'type': row[1], 'value': row[2]})

        # Attach entities to messages
        for msg in messages:
            msg['entities'] = entities_map.get(msg['id'], [])

    conn.close()

    return jsonify({
        'messages': messages,
        'total': total,
        'offset': offset,
        'limit': limit,
        'has_more': offset + limit < total
    })


@app.route('/api/chat/thread/<int:message_id>')
def api_chat_thread(message_id):
    """Get conversation thread for a message."""
    conn = get_db()
    thread = []
    visited = set()

    def get_parent(msg_id):
        """Recursively get parent messages."""
        if msg_id in visited:
            return
        visited.add(msg_id)

        cursor = conn.execute("""
            SELECT id as message_id, date, from_name, text_plain as text, reply_to_message_id
            FROM messages WHERE id = ?
        """, (msg_id,))
        row = cursor.fetchone()

        if row:
            if row['reply_to_message_id']:
                get_parent(row['reply_to_message_id'])
            thread.append(dict(row))

    def get_children(msg_id):
        """Get all replies to a message."""
        cursor = conn.execute("""
            SELECT id as message_id, date, from_name, text_plain as text, reply_to_message_id
            FROM messages WHERE reply_to_message_id = ?
            ORDER BY date
        """, (msg_id,))

        for row in cursor.fetchall():
            if row['message_id'] not in visited:
                visited.add(row['message_id'])
                thread.append(dict(row))
                get_children(row['message_id'])

    # Get the original message and its parents
    get_parent(message_id)

    # Get all replies
    get_children(message_id)

    conn.close()

    # Sort by date
    thread.sort(key=lambda x: x['date'])

    return jsonify(thread)


@app.route('/api/chat/context/<int:message_id>')
def api_chat_context(message_id):
    """Get messages around a specific message."""
    before = int(request.args.get('before', 20))
    after = int(request.args.get('after', 20))

    conn = get_db()

    # Get target message date
    cursor = conn.execute("SELECT date FROM messages WHERE id = ?", (message_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({'messages': [], 'target_id': message_id})

    target_date = row['date']

    # Get messages before
    cursor = conn.execute("""
        SELECT id as message_id, date, from_id, from_name, text_plain as text,
               reply_to_message_id, has_media, has_links as has_link
        FROM messages
        WHERE date < ?
        ORDER BY date DESC
        LIMIT ?
    """, (target_date, before))
    before_msgs = list(reversed([dict(row) for row in cursor.fetchall()]))

    # Get target message
    cursor = conn.execute("""
        SELECT id as message_id, date, from_id, from_name, text_plain as text,
               reply_to_message_id, has_media, has_links as has_link
        FROM messages
        WHERE id = ?
    """, (message_id,))
    target_msg = dict(cursor.fetchone())

    # Get messages after
    cursor = conn.execute("""
        SELECT id as message_id, date, from_id, from_name, text_plain as text,
               reply_to_message_id, has_media, has_links as has_link
        FROM messages
        WHERE date > ?
        ORDER BY date ASC
        LIMIT ?
    """, (target_date, after))
    after_msgs = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return jsonify({
        'messages': before_msgs + [target_msg] + after_msgs,
        'target_id': message_id
    })


# ==========================================
# API ENDPOINTS - AI SEARCH
# ==========================================

# Global AI engine (lazy loaded)
_ai_engine = None
_ai_engine_init_attempted = False

def get_ai_engine():
    """Get or create AI search engine."""
    global _ai_engine, _ai_engine_init_attempted

    if _ai_engine is not None:
        return _ai_engine

    if _ai_engine_init_attempted:
        return None  # Already tried and failed

    _ai_engine_init_attempted = True

    try:
        from ai_search import AISearchEngine
        import os

        provider = os.getenv('AI_PROVIDER', 'ollama')
        # Get API key - check both generic and provider-specific env vars
        api_key = os.getenv('AI_API_KEY') or os.getenv(f'{provider.upper()}_API_KEY')

        print(f"Initializing AI engine with provider: {provider}")
        _ai_engine = AISearchEngine(DB_PATH, provider, api_key)
        print(f"AI engine initialized successfully")
        return _ai_engine
    except Exception as e:
        print(f"AI Search not available: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/ai/status')
def api_ai_status():
    """Get AI engine status for debugging."""
    provider = os.getenv('AI_PROVIDER', 'ollama')
    api_key = os.getenv('AI_API_KEY') or os.getenv(f'{provider.upper()}_API_KEY')

    status = {
        'provider': provider,
        'api_key_set': bool(api_key),
        'api_key_preview': f"{api_key[:8]}..." if api_key and len(api_key) > 8 else None,
        'ai_engine_initialized': _ai_engine is not None,
        'init_attempted': _ai_engine_init_attempted,
        'semantic_search_available': HAS_SEMANTIC_SEARCH,
    }

    # Check if we can initialize now
    if _ai_engine is None and not _ai_engine_init_attempted:
        engine = get_ai_engine()
        status['ai_engine_initialized'] = engine is not None

    # Check for embeddings
    if HAS_SEMANTIC_SEARCH:
        try:
            ss = get_semantic_search()
            status['embeddings_available'] = ss.is_available()
            status['embeddings_stats'] = ss.stats()
        except Exception as e:
            status['embeddings_error'] = str(e)

    return jsonify(status)


@app.route('/api/ai/reset')
def api_ai_reset():
    """Reset AI engine to allow re-initialization."""
    global _ai_engine, _ai_engine_init_attempted
    _ai_engine = None
    _ai_engine_init_attempted = False
    return jsonify({'status': 'reset', 'message': 'AI engine will be reinitialized on next request'})


@app.route('/api/cache/invalidate')
def api_cache_invalidate():
    """Invalidate all caches. Call after DB updates (daily sync, import, etc.)."""
    invalidate_caches()
    return jsonify({'status': 'invalidated', 'new_version': _cache_version})


@app.route('/api/embeddings/reload')
def api_embeddings_reload():
    """Reload embeddings from DB (call after daily sync adds new embeddings)."""
    if not HAS_SEMANTIC_SEARCH:
        return jsonify({'error': 'Semantic search not available'})
    try:
        ss = get_semantic_search()
        old_count = len(ss.message_ids) if ss.embeddings_loaded else 0
        ss.reload_embeddings()
        new_count = len(ss.message_ids)
        return jsonify({
            'status': 'reloaded',
            'previous_count': old_count,
            'new_count': new_count,
            'added': new_count - old_count
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/ai/search', methods=['POST'])
def api_ai_search():
    """AI-powered natural language search."""
    data = request.get_json()
    query = data.get('query', '')
    mode = data.get('mode', 'auto')  # 'auto', 'sql', 'context', or 'semantic'

    if not query:
        return jsonify({'error': 'Query required'})

    # Semantic mode: Use pre-computed embeddings + AI reasoning
    if mode == 'semantic':
        if not HAS_SEMANTIC_SEARCH:
            return jsonify({'error': 'Semantic search not available. Install sentence-transformers.'})
        try:
            ss = get_semantic_search()
            if not ss.is_available():
                return jsonify({'error': 'embeddings.db not found. Run the Colab notebook first.'})

            # Get AI engine for reasoning
            ai_engine = get_ai_engine()

            if ai_engine:
                # Semantic search + AI reasoning
                result = ss.search_with_ai_answer(query, ai_engine, limit=30)
                return jsonify(result)
            else:
                # Just semantic search without AI reasoning
                results = ss.search_with_full_text(query, limit=30)
                provider = os.getenv('AI_PROVIDER', 'ollama')
                api_key_set = bool(os.getenv('AI_API_KEY') or os.getenv(f'{provider.upper()}_API_KEY'))
                return jsonify({
                    'query': query,
                    'mode': 'semantic',
                    'results': results,
                    'count': len(results),
                    'answer': f"נמצאו {len(results)} הודעות דומות סמנטית לשאילתה.\n\n⚠️ AI לא זמין - בדוק שה-API key מוגדר (provider: {provider}, key set: {api_key_set})"
                })
        except Exception as e:
            return jsonify({'error': f'Semantic search error: {str(e)}'})

    engine = get_ai_engine()

    if engine is None:
        # Fallback: Use basic SQL search
        return fallback_ai_search(query)

    try:
        # Context mode: AI reads messages and reasons over them
        if mode == 'context':
            result = engine.context_search(query)
        # SQL mode: Generate SQL and execute
        elif mode == 'sql':
            result = engine.search(query, generate_answer=True)
        # Auto mode: Try SQL first, fall back to context if no results
        else:
            result = engine.search(query, generate_answer=True)
            # If no results or error, try context search
            if result.get('count', 0) == 0 or 'error' in result:
                result = engine.context_search(query)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'query': query})


def fallback_ai_search(query: str):
    """Fallback search when AI is not available."""
    conn = get_db()

    # Simple keyword extraction and search
    keywords = [w for w in query.split() if len(w) > 2]

    if not keywords:
        return jsonify({'error': 'No valid keywords', 'query': query})

    # Build FTS query
    fts_query = ' OR '.join(keywords)

    try:
        cursor = conn.execute('''
            SELECT
                m.id as message_id, m.date, m.from_name, m.text_plain as text
            FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.id
            WHERE messages_fts MATCH ?
            ORDER BY m.date DESC
            LIMIT 20
        ''', (fts_query,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Generate simple answer
        if results:
            answer = f"נמצאו {len(results)} הודעות עם המילים: {', '.join(keywords)}"
        else:
            answer = f"לא נמצאו הודעות עם המילים: {', '.join(keywords)}"

        return jsonify({
            'query': query,
            'sql': f"FTS MATCH: {fts_query}",
            'results': results,
            'count': len(results),
            'answer': answer,
            'fallback': True
        })

    except Exception as e:
        conn.close()
        return jsonify({'error': str(e), 'query': query})


@app.route('/api/ai/thread/<int:message_id>')
def api_ai_thread(message_id):
    """Get full thread using AI-powered analysis."""
    engine = get_ai_engine()

    if engine is None:
        # Use basic thread retrieval
        return api_chat_thread(message_id)

    try:
        thread = engine.get_thread(message_id)
        return jsonify(thread)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/ai/similar/<int:message_id>')
def api_ai_similar(message_id):
    """Find similar messages."""
    limit = int(request.args.get('limit', 10))

    engine = get_ai_engine()

    if engine is None:
        return jsonify({'error': 'AI not available'})

    try:
        similar = engine.find_similar_messages(message_id, limit)
        return jsonify(similar)
    except Exception as e:
        return jsonify({'error': str(e)})


# ==========================================
# API ENDPOINTS - DATABASE UPDATE
# ==========================================

@app.route('/api/update', methods=['POST'])
def api_update_database():
    """
    Update database with new JSON data.

    Accepts JSON file upload or raw JSON in request body.
    Only new messages (not already in DB) will be added.
    """
    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Read and parse JSON
            try:
                json_data = json.loads(file.read().decode('utf-8'))
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
        else:
            # Try to get JSON from request body
            json_data = request.get_json()
            if not json_data:
                return jsonify({'error': 'No JSON data provided'}), 400

        # Import and use IncrementalIndexer
        from indexer import IncrementalIndexer

        indexer = IncrementalIndexer(DB_PATH)
        try:
            stats = indexer.update_from_json_data(json_data, show_progress=False)
        finally:
            indexer.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_in_file': stats['total_in_file'],
                'new_messages': stats['new_messages'],
                'duplicates': stats['duplicates'],
                'entities': stats['entities'],
                'elapsed_seconds': round(stats['elapsed_seconds'], 2)
            }
        })

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/stats')
def api_db_stats():
    """Get database statistics."""
    conn = get_db()

    stats = {}

    # Total messages
    cursor = conn.execute('SELECT COUNT(*) FROM messages')
    stats['total_messages'] = cursor.fetchone()[0]

    # Total users
    cursor = conn.execute('SELECT COUNT(DISTINCT from_id) FROM messages WHERE from_id IS NOT NULL')
    stats['total_users'] = cursor.fetchone()[0]

    # Date range
    cursor = conn.execute('SELECT MIN(date), MAX(date) FROM messages')
    row = cursor.fetchone()
    stats['first_message'] = row[0]
    stats['last_message'] = row[1]

    # Database file size
    import os
    if os.path.exists(DB_PATH):
        stats['db_size_mb'] = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)

    conn.close()

    return jsonify(stats)


# ==========================================
# API ENDPOINTS - EXPORT
# ==========================================

@app.route('/api/export/users')
def api_export_users():
    """Export user data as CSV."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)

    conn = get_db()

    cursor = conn.execute('''
        SELECT
            from_id,
            from_name,
            COUNT(*) as message_count,
            SUM(LENGTH(text_plain)) as char_count,
            SUM(has_links) as links,
            SUM(has_media) as media,
            MIN(date_unixtime) as first_seen,
            MAX(date_unixtime) as last_seen
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        AND from_id IS NOT NULL
        GROUP BY from_id
        ORDER BY message_count DESC
    ''', (start_ts, end_ts))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['User ID', 'Name', 'Messages', 'Characters', 'Links', 'Media', 'First Seen', 'Last Seen'])

    for row in cursor.fetchall():
        writer.writerow([
            row['from_id'],
            row['from_name'],
            row['message_count'],
            row['char_count'] or 0,
            row['links'] or 0,
            row['media'] or 0,
            datetime.fromtimestamp(row['first_seen']).isoformat() if row['first_seen'] else '',
            datetime.fromtimestamp(row['last_seen']).isoformat() if row['last_seen'] else ''
        ])

    conn.close()

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=users_export.csv'}
    )


@app.route('/api/export/messages')
def api_export_messages():
    """Export messages as CSV."""
    timeframe = request.args.get('timeframe', 'all')
    start_ts, end_ts = parse_timeframe(timeframe)
    limit = int(request.args.get('limit', 10000))

    conn = get_db()

    cursor = conn.execute('''
        SELECT
            id, date, from_id, from_name, text_plain,
            has_links, has_media, has_mentions,
            reply_to_message_id
        FROM messages
        WHERE date_unixtime BETWEEN ? AND ?
        ORDER BY date_unixtime DESC
        LIMIT ?
    ''', (start_ts, end_ts, limit))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'User ID', 'User Name', 'Text', 'Has Links', 'Has Media', 'Has Mentions', 'Reply To'])

    for row in cursor.fetchall():
        writer.writerow([
            row['id'],
            row['date'],
            row['from_id'],
            row['from_name'],
            row['text_plain'][:500] if row['text_plain'] else '',
            row['has_links'],
            row['has_media'],
            row['has_mentions'],
            row['reply_to_message_id']
        ])

    conn.close()

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=messages_export.csv'}
    )


# ==========================================
# MAIN
# ==========================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Telegram Analytics Dashboard')
    parser.add_argument('--db', default=os.environ.get('DB_PATH', 'telegram.db'), help='Database path')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Server port')
    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'), help='Server host')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    global DB_PATH
    DB_PATH = args.db

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           TELEGRAM ANALYTICS DASHBOARD                        ║
╠══════════════════════════════════════════════════════════════╣
║  Database: {args.db:47} ║
║  Server:   http://{args.host}:{args.port:<37} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
