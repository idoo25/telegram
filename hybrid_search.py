"""
Hybrid Search - Combines Vector Search, BM25, and Query Expansion

This provides much better search for chat data by:
1. Chunk-based vector search (captures context)
2. BM25 keyword search (finds exact matches)
3. Query expansion (handles variations)
"""

import sqlite3
import numpy as np
import pickle
import re
import os
from typing import List, Dict, Any, Optional

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try importing BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class HybridSearch:
    """
    Hybrid search combining:
    - Chunk-based vector search (conversation context)
    - BM25 keyword search (exact matches)
    - Query expansion (synonyms, variations)
    """

    def __init__(self,
                 messages_db: str = 'telegram.db',
                 chunk_embeddings_db: str = 'chunk_embeddings.db',
                 bm25_index_path: str = 'bm25_index.pkl',
                 single_embeddings_db: str = 'embeddings.db'):
        self.messages_db = messages_db
        self.chunk_embeddings_db = chunk_embeddings_db
        self.bm25_index_path = bm25_index_path
        self.single_embeddings_db = single_embeddings_db

        # Lazy-loaded components
        self.model = None
        self.chunk_embeddings = None
        self.chunk_data = None
        self.bm25 = None
        self.bm25_message_ids = None
        self.single_embeddings = None
        self.single_message_ids = None

    def _load_model(self):
        """Load the embedding model."""
        if self.model is not None:
            return

        if not HAS_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed")

        print("Loading embedding model...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Model loaded!")

    def _load_chunk_embeddings(self):
        """Load chunk embeddings."""
        if self.chunk_embeddings is not None:
            return True

        if not os.path.exists(self.chunk_embeddings_db):
            print(f"Chunk embeddings not found: {self.chunk_embeddings_db}")
            return False

        print(f"Loading chunk embeddings from {self.chunk_embeddings_db}...")
        conn = sqlite3.connect(self.chunk_embeddings_db)

        rows = conn.execute("""
            SELECT chunk_id, text, message_ids, anchor_message_id, embedding
            FROM chunk_embeddings
        """).fetchall()
        conn.close()

        if not rows:
            return False

        import json
        self.chunk_data = []
        emb_list = []

        for row in rows:
            chunk_id, text, msg_ids_json, anchor_id, emb_blob = row
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            emb_list.append(emb)
            self.chunk_data.append({
                'chunk_id': chunk_id,
                'text': text,
                'message_ids': json.loads(msg_ids_json),
                'anchor_message_id': anchor_id
            })

        self.chunk_embeddings = np.vstack(emb_list)
        # Normalize
        norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        self.chunk_embeddings = self.chunk_embeddings / np.where(norms == 0, 1, norms)

        print(f"Loaded {len(self.chunk_data)} chunk embeddings")
        return True

    def _load_single_embeddings(self):
        """Load single-message embeddings (fallback)."""
        if self.single_embeddings is not None:
            return True

        if not os.path.exists(self.single_embeddings_db):
            return False

        print(f"Loading single embeddings from {self.single_embeddings_db}...")
        conn = sqlite3.connect(self.single_embeddings_db)

        rows = conn.execute("""
            SELECT message_id, embedding FROM embeddings
        """).fetchall()
        conn.close()

        if not rows:
            return False

        self.single_message_ids = []
        emb_list = []

        for row in rows:
            msg_id, emb_blob = row
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            emb_list.append(emb)
            self.single_message_ids.append(msg_id)

        self.single_embeddings = np.vstack(emb_list)
        norms = np.linalg.norm(self.single_embeddings, axis=1, keepdims=True)
        self.single_embeddings = self.single_embeddings / np.where(norms == 0, 1, norms)

        print(f"Loaded {len(self.single_message_ids)} single embeddings")
        return True

    def _load_bm25(self):
        """Load BM25 index."""
        if self.bm25 is not None:
            return True

        if not os.path.exists(self.bm25_index_path):
            print(f"BM25 index not found: {self.bm25_index_path}")
            return False

        print(f"Loading BM25 index from {self.bm25_index_path}...")
        with open(self.bm25_index_path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.bm25_message_ids = data['message_ids']
        print(f"Loaded BM25 index with {len(self.bm25_message_ids)} documents")
        return True

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations.
        Returns list of query variations to search.
        """
        queries = [query]

        # Hebrew question word expansions
        expansions = {
            'איפה': ['איפה', 'היכן', 'מיקום', 'כתובת', 'עיר'],
            'מתי': ['מתי', 'באיזה תאריך', 'מועד', 'זמן'],
            'מי': ['מי', 'מיהו', 'מיהי', 'שם'],
            'כמה': ['כמה', 'מספר', 'כמות'],
            'למה': ['למה', 'מדוע', 'סיבה'],
            'גר': ['גר', 'גרה', 'מתגורר', 'מתגוררת', 'גרים'],
            'עובד': ['עובד', 'עובדת', 'עובדים', 'מועסק', 'עבודה'],
        }

        # Add expanded variations
        for word, synonyms in expansions.items():
            if word in query:
                for syn in synonyms:
                    if syn != word:
                        expanded = query.replace(word, syn)
                        if expanded not in queries:
                            queries.append(expanded)

        return queries[:5]  # Limit to 5 variations

    def search_chunks(self, query: str, limit: int = 20) -> List[Dict]:
        """Search using chunk embeddings (context-aware)."""
        if not self._load_chunk_embeddings():
            return []

        self._load_model()

        # Encode query with e5 prefix
        query_emb = self.model.encode([f"query: {query}"], convert_to_numpy=True)[0]
        query_norm = query_emb / np.linalg.norm(query_emb)

        # Compute similarities
        similarities = np.dot(self.chunk_embeddings, query_norm)

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.chunk_data[idx]
            results.append({
                'type': 'chunk',
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'message_ids': chunk['message_ids'],
                'anchor_message_id': chunk['anchor_message_id'],
                'score': score
            })

        return results

    def search_bm25(self, query: str, limit: int = 20) -> List[Dict]:
        """Search using BM25 (keyword-based)."""
        if not self._load_bm25():
            return []

        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top results
        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append({
                    'type': 'bm25',
                    'message_id': self.bm25_message_ids[idx],
                    'score': score
                })

        return results

    def search_single(self, query: str, limit: int = 20) -> List[Dict]:
        """Search using single-message embeddings (fallback)."""
        if not self._load_single_embeddings():
            return []

        self._load_model()

        query_emb = self.model.encode([f"query: {query}"], convert_to_numpy=True)[0]
        query_norm = query_emb / np.linalg.norm(query_emb)

        similarities = np.dot(self.single_embeddings, query_norm)
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append({
                'type': 'single',
                'message_id': self.single_message_ids[idx],
                'score': score
            })

        return results

    def hybrid_search(self, query: str, limit: int = 20,
                      vector_weight: float = 0.6,
                      bm25_weight: float = 0.4,
                      use_expansion: bool = True) -> List[Dict]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            limit: Max results
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            use_expansion: Whether to expand query

        Returns:
            Combined search results
        """
        all_message_scores = {}

        # Get expanded queries
        queries = self.expand_query(query) if use_expansion else [query]

        # Search with each query variation
        for q in queries:
            # Chunk/Vector search
            chunk_results = self.search_chunks(q, limit=limit * 2)
            for r in chunk_results:
                for msg_id in r['message_ids']:
                    if msg_id not in all_message_scores:
                        all_message_scores[msg_id] = {'vector': 0, 'bm25': 0, 'chunk_text': None}
                    # Use max score across message appearances
                    all_message_scores[msg_id]['vector'] = max(
                        all_message_scores[msg_id]['vector'],
                        r['score'] * vector_weight
                    )
                    if all_message_scores[msg_id]['chunk_text'] is None:
                        all_message_scores[msg_id]['chunk_text'] = r['text']

            # BM25 search
            bm25_results = self.search_bm25(q, limit=limit * 2)
            for r in bm25_results:
                msg_id = r['message_id']
                if msg_id not in all_message_scores:
                    all_message_scores[msg_id] = {'vector': 0, 'bm25': 0, 'chunk_text': None}
                all_message_scores[msg_id]['bm25'] = max(
                    all_message_scores[msg_id]['bm25'],
                    r['score'] * bm25_weight / 10  # Normalize BM25 scores
                )

        # Combine scores
        combined = []
        for msg_id, scores in all_message_scores.items():
            total_score = scores['vector'] + scores['bm25']
            combined.append({
                'message_id': msg_id,
                'score': total_score,
                'vector_score': scores['vector'],
                'bm25_score': scores['bm25'],
                'chunk_text': scores['chunk_text']
            })

        # Sort by combined score
        combined.sort(key=lambda x: x['score'], reverse=True)

        return combined[:limit]

    def search_with_context(self, query: str, limit: int = 20,
                           context_window: int = 3) -> List[Dict]:
        """
        Search and return results with surrounding context.

        Args:
            query: Search query
            limit: Max results
            context_window: Messages before/after to include

        Returns:
            Results with full context
        """
        # Get hybrid search results
        results = self.hybrid_search(query, limit=limit)

        if not results:
            return []

        # Get full context from DB
        conn = sqlite3.connect(self.messages_db)
        conn.row_factory = sqlite3.Row

        enriched = []
        for r in results:
            msg_id = r['message_id']

            # Get the message
            msg = conn.execute(
                "SELECT * FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()

            if not msg:
                continue

            # Get surrounding messages
            context_before = conn.execute("""
                SELECT id, date, from_name, text_plain FROM messages
                WHERE date_unixtime < (SELECT date_unixtime FROM messages WHERE id = ?)
                ORDER BY date_unixtime DESC LIMIT ?
            """, (msg_id, context_window)).fetchall()

            context_after = conn.execute("""
                SELECT id, date, from_name, text_plain FROM messages
                WHERE date_unixtime > (SELECT date_unixtime FROM messages WHERE id = ?)
                ORDER BY date_unixtime ASC LIMIT ?
            """, (msg_id, context_window)).fetchall()

            enriched.append({
                'message_id': msg_id,
                'score': r['score'],
                'message': {
                    'id': msg['id'],
                    'date': msg['date'],
                    'from_name': msg['from_name'],
                    'text': msg['text_plain']
                },
                'context_before': [dict(m) for m in reversed(context_before)],
                'context_after': [dict(m) for m in context_after],
                'chunk_text': r.get('chunk_text')
            })

        conn.close()
        return enriched

    def stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        stats = {
            'chunks_available': os.path.exists(self.chunk_embeddings_db),
            'bm25_available': os.path.exists(self.bm25_index_path),
            'single_embeddings_available': os.path.exists(self.single_embeddings_db),
        }

        if stats['chunks_available']:
            conn = sqlite3.connect(self.chunk_embeddings_db)
            stats['chunk_count'] = conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings"
            ).fetchone()[0]
            conn.close()

        if stats['single_embeddings_available']:
            conn = sqlite3.connect(self.single_embeddings_db)
            stats['single_embedding_count'] = conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            conn.close()

        return stats


# Singleton instance
_hybrid_search = None


def get_hybrid_search() -> HybridSearch:
    """Get or create hybrid search instance."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search


# CLI for testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_search.py 'search query'")
        print("\nStats:")
        hs = get_hybrid_search()
        print(hs.stats())
        sys.exit(0)

    query = ' '.join(sys.argv[1:])
    hs = get_hybrid_search()

    print(f"\n=== Searching: {query} ===\n")

    # Show expanded queries
    expanded = hs.expand_query(query)
    print(f"Expanded queries: {expanded}\n")

    # Search
    results = hs.search_with_context(query, limit=5)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.3f}) ---")
        print(f"From: {r['message']['from_name']}")
        print(f"Date: {r['message']['date']}")
        print(f"Text: {r['message']['text'][:200]}...")
        if r['context_before']:
            print(f"\nContext before:")
            for ctx in r['context_before']:
                print(f"  [{ctx['from_name']}] {ctx['text_plain'][:100]}...")
        if r['context_after']:
            print(f"\nContext after:")
            for ctx in r['context_after']:
                print(f"  [{ctx['from_name']}] {ctx['text_plain'][:100]}...")
        print()
