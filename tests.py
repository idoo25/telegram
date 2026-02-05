#!/usr/bin/env python3
"""
Integration tests for Telegram Analytics: indexer, search, and dashboard endpoints.

Run with: python -m pytest tests.py -v
Or:       python tests.py
"""

import json
import os
import sqlite3
import tempfile
import time
import unittest

from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_messages(n: int = 5) -> list[dict]:
    """Generate N realistic Telegram-format messages."""
    base_ts = 1700000000
    users = [
        ("user1", "Alice"),
        ("user2", "Bob"),
        ("user3", "Carol"),
    ]
    msgs = []
    for i in range(1, n + 1):
        uid, name = users[i % len(users)]
        msgs.append({
            "id": 1000 + i,
            "type": "message",
            "date": f"2024-01-{(i % 28) + 1:02d}T10:00:00",
            "date_unixtime": str(base_ts + i * 3600),
            "from": name,
            "from_id": uid,
            "text": f"Test message number {i} from {name}",
            "text_entities": [
                {"type": "plain", "text": f"Test message number {i} from {name}"}
            ],
            "reply_to_message_id": (1000 + i - 1) if i > 1 else None,
        })
    return msgs


def _write_json(path: str, messages: list[dict]):
    """Write messages in Telegram export JSON format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"messages": messages}, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Indexer Tests
# ---------------------------------------------------------------------------

class TestIndexer(unittest.TestCase):
    """Tests for OptimizedIndexer and IncrementalIndexer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.json_path = os.path.join(self.tmpdir, "messages.json")
        self.messages = _sample_messages(10)
        _write_json(self.json_path, self.messages)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_optimized_indexer_indexes_messages(self):
        from indexer import OptimizedIndexer
        indexer = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        stats = indexer.index_file(self.json_path, show_progress=False)

        self.assertGreater(stats["messages"], 0)

        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        self.assertEqual(count, stats["messages"])

    def test_incremental_indexer_deduplication(self):
        from indexer import OptimizedIndexer, IncrementalIndexer

        # First: create DB with OptimizedIndexer
        opt = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        opt.index_file(self.json_path, show_progress=False)

        # Now use IncrementalIndexer – same data, should all be duplicates
        idx = IncrementalIndexer(self.db_path)
        stats = idx.update_from_json(self.json_path, show_progress=False)
        idx.close()

        self.assertEqual(stats["new_messages"], 0)
        self.assertGreater(stats["duplicates"], 0)

    def test_incremental_indexer_adds_new(self):
        from indexer import OptimizedIndexer, IncrementalIndexer

        # Create DB with 5 messages
        msgs5 = _sample_messages(5)
        _write_json(self.json_path, msgs5)
        opt = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        opt.index_file(self.json_path, show_progress=False)

        # Now add 10 messages (5 old + 5 new)
        msgs10 = _sample_messages(10)
        json2 = os.path.join(self.tmpdir, "messages2.json")
        _write_json(json2, msgs10)

        idx = IncrementalIndexer(self.db_path)
        stats = idx.update_from_json(json2, show_progress=False)
        idx.close()

        self.assertEqual(stats["new_messages"], 5)
        self.assertEqual(stats["duplicates"], 5)

    def test_incremental_indexer_from_json_data(self):
        from indexer import OptimizedIndexer, IncrementalIndexer

        # Init DB first
        opt = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        opt.index_file(self.json_path, show_progress=False)

        # Add new messages via json_data
        new_msgs = _sample_messages(15)  # 10 old + 5 new
        idx = IncrementalIndexer(self.db_path)
        stats = idx.update_from_json_data(new_msgs, show_progress=False)
        idx.close()

        self.assertEqual(stats["new_messages"], 5)

    def test_fts5_search_works(self):
        from indexer import OptimizedIndexer
        indexer = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        indexer.index_file(self.json_path, show_progress=False)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'message'"
        )
        count = cursor.fetchone()[0]
        conn.close()

        self.assertGreater(count, 0, "FTS5 search should find messages with 'message'")

    def test_streaming_load_json_messages(self):
        from indexer import load_json_messages
        msgs = list(load_json_messages(self.json_path))
        self.assertEqual(len(msgs), 10)
        self.assertIn("text_plain", msgs[0])

    def test_entities_extracted(self):
        """Messages with links/mentions in text_entities should have entities stored."""
        msgs = [
            {
                "id": 9001,
                "type": "message",
                "date": "2024-01-01T10:00:00",
                "date_unixtime": "1700000000",
                "from": "Alice",
                "from_id": "user1",
                "text": "Check https://example.com and @bob",
                "text_entities": [
                    {"type": "plain", "text": "Check "},
                    {"type": "link", "text": "https://example.com"},
                    {"type": "plain", "text": " and "},
                    {"type": "mention", "text": "@bob"},
                ],
            }
        ]
        _write_json(self.json_path, msgs)

        from indexer import OptimizedIndexer
        indexer = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        indexer.index_file(self.json_path, show_progress=False)

        conn = sqlite3.connect(self.db_path)
        entities = conn.execute("SELECT type, value FROM entities WHERE message_id = 9001").fetchall()
        conn.close()

        types = [e[0] for e in entities]
        self.assertIn("link", types)
        self.assertIn("mention", types)


# ---------------------------------------------------------------------------
# 2. Search Tests
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):
    """Tests for FTS search."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.json_path = os.path.join(self.tmpdir, "messages.json")
        _write_json(self.json_path, _sample_messages(20))

        from indexer import OptimizedIndexer
        indexer = OptimizedIndexer(self.db_path, build_trigrams=False, build_graph=False)
        indexer.index_file(self.json_path, show_progress=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fts_match_query(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, text_plain FROM messages WHERE id IN "
            "(SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'Alice')"
        ).fetchall()
        conn.close()
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn("Alice", r["text_plain"])

    def test_fts_returns_no_results_for_nonsense(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'xyzzyplugh'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(rows, 0)


# ---------------------------------------------------------------------------
# 3. SemanticSearch Empty Embeddings
# ---------------------------------------------------------------------------

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestSemanticSearchEmpty(unittest.TestCase):
    """Test that SemanticSearch handles missing/empty embeddings gracefully."""

    def test_is_available_missing_db(self):
        from semantic_search import SemanticSearch
        ss = SemanticSearch(embeddings_db="/tmp/nonexistent_embeddings_12345.db")
        self.assertFalse(ss.is_available())

    def test_is_available_empty_db(self):
        from semantic_search import SemanticSearch
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "empty_emb.db")

        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE embeddings (message_id INTEGER PRIMARY KEY, "
            "from_name TEXT, text_preview TEXT, embedding BLOB)"
        )
        conn.commit()
        conn.close()

        ss = SemanticSearch(embeddings_db=db_path)
        self.assertFalse(ss.is_available())

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_empty_embeddings_no_crash(self):
        from semantic_search import SemanticSearch
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "empty_emb.db")

        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE embeddings (message_id INTEGER PRIMARY KEY, "
            "from_name TEXT, text_preview TEXT, embedding BLOB)"
        )
        conn.commit()
        conn.close()

        ss = SemanticSearch(embeddings_db=db_path)
        ss._load_embeddings()  # Should not crash

        self.assertTrue(ss.embeddings_loaded)
        self.assertEqual(len(ss.message_ids), 0)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stats_empty_db(self):
        from semantic_search import SemanticSearch
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "empty_emb.db")

        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE embeddings (message_id INTEGER PRIMARY KEY, "
            "from_name TEXT, text_preview TEXT, embedding BLOB)"
        )
        conn.commit()
        conn.close()

        ss = SemanticSearch(embeddings_db=db_path)
        s = ss.stats()
        self.assertTrue(s["available"])  # File exists and table exists
        self.assertEqual(s["count"], 0)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Dashboard Endpoint Tests
# ---------------------------------------------------------------------------

try:
    import flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@unittest.skipUnless(HAS_FLASK, "flask not installed")
class TestDashboardEndpoints(unittest.TestCase):
    """Test Flask dashboard API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Create a test DB and configure Flask test client."""
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "test.db")
        cls.json_path = os.path.join(cls.tmpdir, "messages.json")

        _write_json(cls.json_path, _sample_messages(50))

        from indexer import OptimizedIndexer
        indexer = OptimizedIndexer(cls.db_path, build_trigrams=False, build_graph=False)
        indexer.index_file(cls.json_path, show_progress=False)

        import dashboard
        dashboard.DB_PATH = cls.db_path
        dashboard.app.config["TESTING"] = True
        cls.client = dashboard.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_overview_endpoint(self):
        resp = self.client.get("/api/overview?timeframe=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_messages", data)
        self.assertGreater(data["total_messages"], 0)

    def test_users_endpoint(self):
        resp = self.client.get("/api/users?timeframe=all&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("users", data)
        self.assertGreater(len(data["users"]), 0)
        user = data["users"][0]
        for field in ("user_id", "name", "messages", "percentage"):
            self.assertIn(field, user)

    def test_users_include_inactive(self):
        resp = self.client.get("/api/users?timeframe=all&include_inactive=0")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for user in data["users"]:
            self.assertGreater(user["messages"], 0)

    def test_search_fts_endpoint(self):
        resp = self.client.get("/api/search?q=message&mode=fts&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("results", data)

    def test_chart_hourly_endpoint(self):
        resp = self.client.get("/api/chart/hourly?timeframe=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 24)

    def test_chart_daily_endpoint(self):
        resp = self.client.get("/api/chart/daily?timeframe=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)

    def test_cache_invalidate_endpoint(self):
        resp = self.client.get("/api/cache/invalidate")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "invalidated")

    def test_page_routes_return_200(self):
        """All page routes should return 200."""
        for route in ("/", "/users", "/search", "/chat", "/moderation", "/settings"):
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed")

    def test_user_profile_endpoint(self):
        resp = self.client.get("/api/users?timeframe=all&limit=1")
        data = resp.get_json()
        if data["users"]:
            uid = data["users"][0]["user_id"]
            resp2 = self.client.get(f"/api/user/{uid}/profile")
            self.assertEqual(resp2.status_code, 200)
            profile = resp2.get_json()
            self.assertIn("total_messages", profile)
            self.assertIn("hourly_activity", profile)

    def test_overview_has_expected_keys(self):
        resp = self.client.get("/api/overview?timeframe=all")
        data = resp.get_json()
        for key in ("total_messages", "total_users", "total_links", "total_media"):
            self.assertIn(key, data, f"Missing key: {key}")


# ---------------------------------------------------------------------------
# 5. AI Search Schema Test
# ---------------------------------------------------------------------------

class TestAISearchSchema(unittest.TestCase):
    """Test that AI search schema generation matches actual DB."""

    def test_dynamic_schema_includes_real_columns(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")

        # Initialize DB with real schema
        from indexer import init_database
        conn = init_database(db_path)
        conn.close()

        from ai_search import AISearchEngine
        # Create instance without connecting to a provider
        engine = AISearchEngine.__new__(AISearchEngine)
        engine.db_path = db_path

        schema = engine._get_db_schema()

        # Verify real column names are present
        self.assertIn("text_plain", schema)
        self.assertIn("date_unixtime", schema)
        self.assertIn("has_links", schema)
        self.assertIn("has_media", schema)
        self.assertIn("from_id", schema)
        self.assertIn("participants", schema)

        # Verify old wrong column names are NOT in the dynamic output
        self.assertNotIn("char_count", schema)
        # media_type would not appear unless there's a column named that
        lines_lower = schema.lower()
        # "media_type" should not be a column name (has_media is the real one)
        self.assertNotIn("media_type (", lines_lower)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
