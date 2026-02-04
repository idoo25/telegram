"""
Semantic Search using pre-computed embeddings from Colab.
Lightweight - only needs sentence-transformers for query encoding.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    SentenceTransformer = None


class SemanticSearch:
    """
    Semantic search using pre-computed embeddings.

    The embeddings.db file is created by running the Colab notebook.
    This class just loads and searches them.
    """

    def __init__(self, embeddings_db: str = 'embeddings.db', messages_db: str = 'telegram.db'):
        self.embeddings_db = embeddings_db
        self.messages_db = messages_db
        self.model = None
        self.embeddings_loaded = False
        self.embeddings = []
        self.message_ids = []
        self.from_names = []
        self.text_previews = []

    def _load_model(self):
        """Load the embedding model (same one used in Colab)."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError(
                "sentence-transformers not installed.\n"
                "Install with: pip install sentence-transformers"
            )
        if self.model is None:
            print("Loading embedding model...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Model loaded!")

    def _load_embeddings(self):
        """Load all embeddings into memory for fast search."""
        if self.embeddings_loaded:
            return

        print(f"Loading embeddings from {self.embeddings_db}...")
        conn = sqlite3.connect(self.embeddings_db)
        cursor = conn.execute(
            "SELECT message_id, from_name, text_preview, embedding FROM embeddings"
        )

        for row in cursor:
            msg_id, name, text, emb_blob = row
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            self.message_ids.append(msg_id)
            self.from_names.append(name or '')
            self.text_previews.append(text or '')
            self.embeddings.append(emb)

        conn.close()

        # Stack into numpy array for fast computation
        self.embeddings = np.vstack(self.embeddings)
        self.embeddings_loaded = True
        print(f"Loaded {len(self.message_ids)} embeddings")

    def search(self, query: str, limit: int = 50, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for semantically similar messages.

        Args:
            query: The search query
            limit: Max results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of dicts with message_id, from_name, text, score
        """
        self._load_model()
        self._load_embeddings()

        # Encode query
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]

        # Compute cosine similarity with all embeddings
        # embeddings are already normalized from Colab
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(self.embeddings, query_norm)

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit * 2]  # Get more, then filter

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue
            results.append({
                'message_id': int(self.message_ids[idx]),
                'from_name': self.from_names[idx],
                'text': self.text_previews[idx],
                'score': score
            })
            if len(results) >= limit:
                break

        return results

    def search_with_full_text(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search and return full message text from messages DB.
        """
        results = self.search(query, limit=limit)

        if not results:
            return []

        # Get full text from messages DB
        conn = sqlite3.connect(self.messages_db)
        conn.row_factory = sqlite3.Row

        for result in results:
            cursor = conn.execute(
                "SELECT date, from_name, text_plain FROM messages WHERE id = ?",
                (result['message_id'],)
            )
            row = cursor.fetchone()
            if row:
                result['date'] = row['date']
                result['from_name'] = row['from_name']
                result['text'] = row['text_plain']

        conn.close()
        return results

    def is_available(self) -> bool:
        """Check if semantic search is available."""
        import os
        return HAS_TRANSFORMERS and os.path.exists(self.embeddings_db)

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        import os

        if not os.path.exists(self.embeddings_db):
            return {'available': False, 'error': 'embeddings.db not found'}

        conn = sqlite3.connect(self.embeddings_db)
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()

        size_mb = os.path.getsize(self.embeddings_db) / (1024 * 1024)

        return {
            'available': True,
            'count': count,
            'size_mb': round(size_mb, 1),
            'model': 'paraphrase-multilingual-MiniLM-L12-v2'
        }


# Singleton instance
_search_instance = None

def get_semantic_search() -> SemanticSearch:
    """Get or create semantic search instance."""
    global _search_instance
    if _search_instance is None:
        _search_instance = SemanticSearch()
    return _search_instance


if __name__ == '__main__':
    # Test
    ss = SemanticSearch()
    print("Stats:", ss.stats())

    if ss.is_available():
        results = ss.search("איפה אתה עובד?", limit=5)
        print("\nResults for 'איפה אתה עובד?':")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['from_name']}: {r['text'][:60]}...")
