#!/usr/bin/env python3
"""
Vector Search Module for Semantic Similarity

Optional module that adds semantic search capabilities using:
- Sentence embeddings (sentence-transformers)
- FAISS for efficient similarity search

Dependencies (optional, install with):
    pip install sentence-transformers faiss-cpu numpy

If dependencies are not installed, the module gracefully degrades.
"""

import sqlite3
import pickle
from pathlib import Path
from typing import Optional

# Try importing optional dependencies
VECTOR_SEARCH_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

VECTOR_SEARCH_AVAILABLE = all([NUMPY_AVAILABLE, FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE])


class VectorSearchUnavailable:
    """Placeholder when dependencies are not installed."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise RuntimeError(
                "Vector search requires additional dependencies. Install with:\n"
                "pip install sentence-transformers faiss-cpu numpy"
            )
        return method


class VectorSearch:
    """
    Semantic search using sentence embeddings and FAISS.

    Features:
    - Generate embeddings for messages
    - Build FAISS index for fast similarity search
    - Find semantically similar messages (not just keyword match)
    - Supports Hebrew and multilingual text

    Example:
        vs = VectorSearch(db_path='telegram.db')
        vs.build_index()  # One-time, can take a while

        # Find similar messages
        results = vs.search("מה קורה היום?", limit=10)
        for msg_id, score, text in results:
            print(f"{score:.3f}: {text[:50]}")
    """

    # Recommended models for multilingual/Hebrew support
    MODELS = {
        'fast': 'paraphrase-multilingual-MiniLM-L12-v2',  # Fast, good multilingual
        'accurate': 'paraphrase-multilingual-mpnet-base-v2',  # More accurate
        'small': 'all-MiniLM-L6-v2',  # Smallest, English-focused
    }

    def __init__(
        self,
        db_path: str = 'telegram.db',
        model_name: str = 'fast',
        index_path: Optional[str] = None
    ):
        """
        Initialize vector search.

        Args:
            db_path: Path to SQLite database
            model_name: Model preset ('fast', 'accurate', 'small') or full model name
            index_path: Path to save/load FAISS index (default: db_path + '.faiss')
        """
        if not VECTOR_SEARCH_AVAILABLE:
            raise RuntimeError(
                "Vector search requires additional dependencies. Install with:\n"
                "pip install sentence-transformers faiss-cpu numpy"
            )

        self.db_path = db_path
        self.index_path = index_path or f"{db_path}.faiss"
        self.id_map_path = f"{self.index_path}.ids"

        # Load model
        model_id = self.MODELS.get(model_name, model_name)
        print(f"Loading embedding model: {model_id}")
        self.model = SentenceTransformer(model_id)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = None
        self.id_map: list[int] = []  # Maps FAISS index position to message_id

        # Try to load existing index
        if Path(self.index_path).exists():
            self.load_index()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> 'np.ndarray':
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of shape (n_texts, dimension)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

    def build_index(
        self,
        batch_size: int = 1000,
        min_text_length: int = 10,
        use_gpu: bool = False
    ) -> None:
        """
        Build FAISS index from all messages in database.

        Args:
            batch_size: Number of messages to process at once
            min_text_length: Minimum text length to index
            use_gpu: Use GPU acceleration if available
        """
        conn = self._get_connection()

        # Count messages
        cursor = conn.execute(
            'SELECT COUNT(*) FROM messages WHERE length(text_plain) >= ?',
            (min_text_length,)
        )
        total = cursor.fetchone()[0]
        print(f"Building index for {total} messages...")

        # Create FAISS index
        # Using IndexFlatIP (Inner Product) since we normalize embeddings
        self.index = faiss.IndexFlatIP(self.dimension)

        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU acceleration")
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )

        self.id_map = []

        # Process in batches
        offset = 0
        while offset < total:
            cursor = conn.execute(
                '''
                SELECT id, text_plain FROM messages
                WHERE length(text_plain) >= ?
                ORDER BY id
                LIMIT ? OFFSET ?
                ''',
                (min_text_length, batch_size, offset)
            )

            rows = cursor.fetchall()
            if not rows:
                break

            ids = [row['id'] for row in rows]
            texts = [row['text_plain'] for row in rows]

            # Encode batch
            embeddings = self.encode(texts, show_progress=False)

            # Add to index
            self.index.add(embeddings)
            self.id_map.extend(ids)

            offset += len(rows)
            print(f"Indexed {offset}/{total} messages ({100*offset/total:.1f}%)")

        conn.close()

        # Save index
        self.save_index()
        print(f"Index built: {self.index.ntotal} vectors")

    def save_index(self) -> None:
        """Save FAISS index and ID map to disk."""
        if self.index is None:
            return

        # Convert GPU index to CPU for saving
        if hasattr(faiss, 'index_gpu_to_cpu'):
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            except:
                cpu_index = self.index
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, self.index_path)

        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)

        print(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0
    ) -> list[tuple[int, float, str]]:
        """
        Search for semantically similar messages.

        Args:
            query: Search query text
            limit: Maximum results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of (message_id, score, text) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Encode query
        query_vector = self.encode([query], show_progress=False)

        # Search FAISS
        scores, indices = self.index.search(query_vector, limit)

        # Get message texts from DB
        conn = self._get_connection()
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue

            message_id = self.id_map[idx]
            cursor = conn.execute(
                'SELECT text_plain FROM messages WHERE id = ?',
                (message_id,)
            )
            row = cursor.fetchone()
            if row:
                results.append((message_id, float(score), row['text_plain']))

        conn.close()
        return results

    def find_similar(
        self,
        message_id: int,
        limit: int = 10,
        exclude_same_user: bool = False
    ) -> list[tuple[int, float, str]]:
        """
        Find messages similar to a specific message.

        Args:
            message_id: ID of the reference message
            limit: Maximum results to return
            exclude_same_user: Exclude messages from same user

        Returns:
            List of (message_id, score, text) tuples
        """
        conn = self._get_connection()

        # Get the reference message
        cursor = conn.execute(
            'SELECT text_plain, from_id FROM messages WHERE id = ?',
            (message_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []

        reference_text = row['text_plain']
        reference_user = row['from_id']
        conn.close()

        # Search
        results = self.search(reference_text, limit=limit * 2)

        # Filter
        filtered = []
        for msg_id, score, text in results:
            if msg_id == message_id:
                continue
            if exclude_same_user:
                conn = self._get_connection()
                cursor = conn.execute(
                    'SELECT from_id FROM messages WHERE id = ?',
                    (msg_id,)
                )
                msg_row = cursor.fetchone()
                conn.close()
                if msg_row and msg_row['from_id'] == reference_user:
                    continue
            filtered.append((msg_id, score, text))
            if len(filtered) >= limit:
                break

        return filtered

    def cluster_messages(
        self,
        n_clusters: int = 10,
        sample_size: Optional[int] = None
    ) -> dict[int, list[int]]:
        """
        Cluster messages by semantic similarity using K-means.

        Args:
            n_clusters: Number of clusters
            sample_size: Number of messages to sample (None = all)

        Returns:
            Dict mapping cluster_id to list of message_ids
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Get vectors
        n_vectors = self.index.ntotal
        if sample_size and sample_size < n_vectors:
            indices = np.random.choice(n_vectors, sample_size, replace=False)
            vectors = np.array([self.index.reconstruct(int(i)) for i in indices])
            ids = [self.id_map[i] for i in indices]
        else:
            vectors = np.array([self.index.reconstruct(i) for i in range(n_vectors)])
            ids = self.id_map

        # K-means clustering
        kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=True)
        kmeans.train(vectors)

        # Assign clusters
        _, assignments = kmeans.index.search(vectors, 1)

        # Group by cluster
        clusters: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
        for msg_id, cluster_id in zip(ids, assignments.flatten()):
            clusters[int(cluster_id)].append(msg_id)

        return clusters

    @property
    def stats(self) -> dict:
        """Get index statistics."""
        return {
            'available': VECTOR_SEARCH_AVAILABLE,
            'model': self.model.get_sentence_embedding_dimension() if self.model else None,
            'dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0,
            'index_path': self.index_path
        }


# Export appropriate class based on availability
if VECTOR_SEARCH_AVAILABLE:
    SemanticSearch = VectorSearch
else:
    SemanticSearch = VectorSearchUnavailable


def check_dependencies() -> dict:
    """Check which dependencies are available."""
    return {
        'numpy': NUMPY_AVAILABLE,
        'faiss': FAISS_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
        'vector_search_available': VECTOR_SEARCH_AVAILABLE
    }


if __name__ == '__main__':
    print("=== Vector Search Dependencies ===")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    if VECTOR_SEARCH_AVAILABLE:
        print("\nVector search is available!")
        print("Usage:")
        print("  vs = VectorSearch('telegram.db')")
        print("  vs.build_index()  # One-time indexing")
        print("  results = vs.search('מה קורה?')")
    else:
        print("\nTo enable vector search, install dependencies:")
        print("  pip install sentence-transformers faiss-cpu numpy")
