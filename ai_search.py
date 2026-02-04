"""
AI-Powered Search for Telegram Analytics
Supports: Ollama (local), Groq (free API), Google Gemini (free API)
"""

import sqlite3
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# Try to import AI libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class AISearchEngine:
    """AI-powered natural language search for Telegram messages."""

    def __init__(self, db_path: str, provider: str = "ollama", api_key: str = None):
        """
        Initialize AI search engine.

        Args:
            db_path: Path to SQLite database
            provider: "ollama", "groq", or "gemini"
            api_key: API key for Groq or Gemini (not needed for Ollama)
        """
        self.db_path = db_path
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        # Initialize provider
        if provider == "groq" and HAS_GROQ:
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.1-70b-versatile"
        elif provider == "gemini" and HAS_GEMINI:
            genai.configure(api_key=self.api_key)
            # Using 2.5 Flash - free, fast, with reasoning capabilities
            self.client = genai.GenerativeModel("gemini-2.5-flash")
        elif provider == "ollama":
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        else:
            raise ValueError(f"Provider {provider} not available. Install required packages.")

    def _get_db_schema(self) -> str:
        """Get database schema for context."""
        return """
        Database Schema:
        - messages: id, message_id, date, from_id, from_name, text, reply_to_message_id,
                   forwarded_from, media_type, has_link, char_count
        - messages_fts: Full-text search on text content (use MATCH for search)
        - users: user_id, name (aggregated from messages)

        Key columns:
        - date: ISO format datetime (e.g., '2024-01-15T14:30:00')
        - from_name: User's display name
        - reply_to_message_id: ID of message being replied to (NULL if not a reply)
        - has_link: 1 if message contains URL, 0 otherwise
        - media_type: 'photo', 'video', 'document', etc. or NULL
        """

    def _get_sample_data(self) -> str:
        """Get sample data for context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get user list
        cursor.execute("""
            SELECT from_name, COUNT(*) as cnt
            FROM messages
            WHERE from_name IS NOT NULL
            GROUP BY from_name
            ORDER BY cnt DESC
            LIMIT 10
        """)
        users = cursor.fetchall()

        # Get date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM messages")
        date_range = cursor.fetchone()

        conn.close()

        return f"""
        Top users: {', '.join([u[0] for u in users])}
        Date range: {date_range[0]} to {date_range[1]}
        """

    def _build_prompt(self, user_query: str) -> str:
        """Build prompt for AI model."""
        schema = self._get_db_schema()
        sample = self._get_sample_data()

        return f"""You are a SQL query generator for a Telegram chat database.
Your task is to convert natural language questions into SQLite queries.

{schema}

{sample}

IMPORTANT RULES:
1. Return ONLY valid SQLite query, no explanations
2. For text search, use: SELECT * FROM messages WHERE id IN (SELECT id FROM messages_fts WHERE messages_fts MATCH 'search_term')
3. For Hebrew text, the FTS5 will handle it correctly
4. Always include relevant columns like date, from_name, text
5. Limit results to 50 unless specified
6. For "who" questions, GROUP BY from_name and COUNT(*)
7. For "when" questions, include date in SELECT
8. For threads/replies, JOIN messages m2 ON m1.reply_to_message_id = m2.message_id

User question: {user_query}

SQLite query:"""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for Ollama")

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        response = self.client.generate_content(prompt)
        return response.text

    def _generate_sql(self, user_query: str) -> str:
        """Generate SQL from natural language query."""
        prompt = self._build_prompt(user_query)

        if self.provider == "ollama":
            response = self._call_ollama(prompt)
        elif self.provider == "groq":
            response = self._call_groq(prompt)
        elif self.provider == "gemini":
            response = self._call_gemini(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Extract SQL from response
        sql = response.strip()

        # Clean up common issues
        sql = re.sub(r'^```sql\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        sql = sql.strip()

        # Ensure it's a SELECT query for safety
        if not sql.upper().startswith("SELECT"):
            raise ValueError("AI generated non-SELECT query")

        return sql

    def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dicts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
        except sqlite3.Error as e:
            results = [{"error": str(e), "sql": sql}]
        finally:
            conn.close()

        return results

    def _generate_answer(self, user_query: str, results: List[Dict], sql: str) -> str:
        """Generate natural language answer from results."""
        if not results:
            return "לא נמצאו תוצאות."

        if "error" in results[0]:
            return f"שגיאה בשאילתה: {results[0]['error']}"

        # Build answer prompt
        results_str = json.dumps(results[:20], ensure_ascii=False, indent=2)

        answer_prompt = f"""Based on the following query results, provide a concise answer in Hebrew.

User question: {user_query}

Query results (JSON):
{results_str}

Total results: {len(results)}

Provide a helpful, concise answer in Hebrew. Include specific names, dates, and numbers from the results.
If showing a list, format it nicely. Keep it brief but informative."""

        if self.provider == "ollama":
            answer = self._call_ollama(answer_prompt)
        elif self.provider == "groq":
            answer = self._call_groq(answer_prompt)
        elif self.provider == "gemini":
            answer = self._call_gemini(answer_prompt)

        return answer

    def search(self, query: str, generate_answer: bool = True) -> Dict[str, Any]:
        """
        Perform AI-powered search.

        Args:
            query: Natural language question in Hebrew or English
            generate_answer: Whether to generate natural language answer

        Returns:
            Dict with sql, results, and optionally answer
        """
        try:
            # Generate SQL
            sql = self._generate_sql(query)

            # Execute query
            results = self._execute_sql(sql)

            response = {
                "query": query,
                "sql": sql,
                "results": results,
                "count": len(results)
            }

            # Generate natural language answer
            if generate_answer and results and "error" not in results[0]:
                response["answer"] = self._generate_answer(query, results, sql)

            return response

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "results": []
            }

    def get_thread(self, message_id: int) -> List[Dict[str, Any]]:
        """Get full conversation thread for a message."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        thread = []
        visited = set()

        def get_parent(msg_id):
            """Recursively get parent messages."""
            if msg_id in visited:
                return
            visited.add(msg_id)

            cursor.execute("""
                SELECT message_id, date, from_name, text, reply_to_message_id
                FROM messages WHERE message_id = ?
            """, (msg_id,))
            row = cursor.fetchone()

            if row:
                if row['reply_to_message_id']:
                    get_parent(row['reply_to_message_id'])
                thread.append(dict(row))

        def get_children(msg_id):
            """Get all replies to a message."""
            cursor.execute("""
                SELECT message_id, date, from_name, text, reply_to_message_id
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

        return thread

    def find_similar_messages(self, message_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Find messages similar to the given message using trigrams."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get the original message
        cursor.execute("SELECT text FROM messages WHERE message_id = ?", (message_id,))
        row = cursor.fetchone()

        if not row or not row['text']:
            return []

        # Use FTS5 to find similar messages
        words = row['text'].split()[:5]  # Use first 5 words
        search_term = ' OR '.join(words)

        cursor.execute("""
            SELECT m.message_id, m.date, m.from_name, m.text
            FROM messages m
            WHERE m.id IN (
                SELECT id FROM messages_fts
                WHERE messages_fts MATCH ?
            )
            AND m.message_id != ?
            LIMIT ?
        """, (search_term, message_id, limit))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results


class ChatViewer:
    """View chat messages like Telegram."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_messages(self,
                     offset: int = 0,
                     limit: int = 50,
                     user_id: str = None,
                     search: str = None,
                     date_from: str = None,
                     date_to: str = None,
                     has_media: bool = None,
                     has_link: bool = None) -> Dict[str, Any]:
        """
        Get messages with Telegram-like pagination.

        Returns messages in reverse chronological order (newest first).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        conditions = []
        params = []

        if user_id:
            conditions.append("from_id = ?")
            params.append(user_id)

        if date_from:
            conditions.append("date >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("date <= ?")
            params.append(date_to)

        if has_media is not None:
            if has_media:
                conditions.append("media_type IS NOT NULL")
            else:
                conditions.append("media_type IS NULL")

        if has_link is not None:
            conditions.append("has_link = ?")
            params.append(1 if has_link else 0)

        # Handle search
        if search:
            conditions.append("""id IN (
                SELECT id FROM messages_fts WHERE messages_fts MATCH ?
            )""")
            params.append(search)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM messages WHERE {where_clause}", params)
        total = cursor.fetchone()[0]

        # Get messages
        query = f"""
            SELECT
                m.message_id,
                m.date,
                m.from_id,
                m.from_name,
                m.text,
                m.reply_to_message_id,
                m.forwarded_from,
                m.media_type,
                m.has_link,
                m.char_count,
                r.from_name as reply_to_name,
                r.text as reply_to_text
            FROM messages m
            LEFT JOIN messages r ON m.reply_to_message_id = r.message_id
            WHERE {where_clause}
            ORDER BY m.date DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor.execute(query, params)
        messages = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "messages": messages,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total
        }

    def get_message_context(self, message_id: int, before: int = 10, after: int = 10) -> Dict[str, Any]:
        """Get messages around a specific message (for context view)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get the target message date
        cursor.execute("SELECT date FROM messages WHERE message_id = ?", (message_id,))
        row = cursor.fetchone()

        if not row:
            return {"messages": [], "target_id": message_id}

        target_date = row['date']

        # Get messages before
        cursor.execute("""
            SELECT message_id, date, from_id, from_name, text,
                   reply_to_message_id, media_type, has_link
            FROM messages
            WHERE date < ?
            ORDER BY date DESC
            LIMIT ?
        """, (target_date, before))
        before_msgs = list(reversed([dict(row) for row in cursor.fetchall()]))

        # Get target message
        cursor.execute("""
            SELECT message_id, date, from_id, from_name, text,
                   reply_to_message_id, media_type, has_link
            FROM messages
            WHERE message_id = ?
        """, (message_id,))
        target_msg = dict(cursor.fetchone())

        # Get messages after
        cursor.execute("""
            SELECT message_id, date, from_id, from_name, text,
                   reply_to_message_id, media_type, has_link
            FROM messages
            WHERE date > ?
            ORDER BY date ASC
            LIMIT ?
        """, (target_date, after))
        after_msgs = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "messages": before_msgs + [target_msg] + after_msgs,
            "target_id": message_id
        }

    def get_user_conversation(self, user1_id: str, user2_id: str, limit: int = 100) -> List[Dict]:
        """Get conversation between two users (their replies to each other)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT m1.message_id, m1.date, m1.from_id, m1.from_name, m1.text,
                   m1.reply_to_message_id, m2.from_name as reply_to_name
            FROM messages m1
            LEFT JOIN messages m2 ON m1.reply_to_message_id = m2.message_id
            WHERE (m1.from_id = ? AND m2.from_id = ?)
               OR (m1.from_id = ? AND m2.from_id = ?)
            ORDER BY m1.date DESC
            LIMIT ?
        """, (user1_id, user2_id, user2_id, user1_id, limit))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI-powered Telegram search")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--api-key", help="API key for cloud providers")

    args = parser.parse_args()

    if args.query:
        engine = AISearchEngine(args.db, args.provider, args.api_key)
        result = engine.search(args.query)

        print(f"\nQuery: {result['query']}")
        print(f"SQL: {result.get('sql', 'N/A')}")
        print(f"Results: {result.get('count', 0)}")

        if 'answer' in result:
            print(f"\nAnswer:\n{result['answer']}")

        if result.get('results'):
            print(f"\nFirst 3 results:")
            for r in result['results'][:3]:
                print(json.dumps(r, ensure_ascii=False, indent=2))
