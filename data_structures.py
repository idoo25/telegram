#!/usr/bin/env python3
"""
Advanced Data Structures for Efficient Search and Traversal

Includes:
- Bloom Filter: O(1) "definitely not in set" checks
- Trie: O(k) prefix search and autocomplete
- LRU Cache: O(1) cached query results
- Graph algorithms: DFS, BFS for thread traversal
"""

import hashlib
import math
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Generator, Iterator, Optional
from functools import wraps


# ============================================
# BLOOM FILTER
# ============================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership testing.

    - O(k) insert and lookup where k is number of hash functions
    - False positives possible, false negatives impossible
    - Use case: Quick "message ID exists?" check before DB query

    Example:
        bf = BloomFilter(expected_items=100000, fp_rate=0.01)
        bf.add("message_123")
        if "message_123" in bf:  # O(1) check
            # Might exist, check DB
        else:
            # Definitely doesn't exist, skip DB
    """

    def __init__(self, expected_items: int = 100000, fp_rate: float = 0.01):
        """
        Initialize Bloom filter.

        Args:
            expected_items: Expected number of items to store
            fp_rate: Desired false positive rate (0.01 = 1%)
        """
        # Calculate optimal size and hash count
        self.size = self._optimal_size(expected_items, fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        self.bit_array = bytearray(math.ceil(self.size / 8))
        self.count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size: m = -n*ln(p) / (ln2)^2"""
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal hash count: k = (m/n) * ln2"""
        return max(1, int((m / n) * math.log(2)))

    def _get_hash_values(self, item: str) -> Generator[int, None, None]:
        """Generate k hash values using double hashing technique."""
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        """Add an item to the filter. O(k) where k is hash count."""
        for pos in self._get_hash_values(item):
            byte_idx, bit_idx = divmod(pos, 8)
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        """Check if item might be in the filter. O(k)."""
        for pos in self._get_hash_values(item):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False  # Definitely not in set
        return True  # Might be in set

    def __len__(self) -> int:
        return self.count

    @property
    def memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return len(self.bit_array)


# ============================================
# TRIE (PREFIX TREE)
# ============================================

class TrieNode:
    """Node in a Trie data structure."""
    __slots__ = ['children', 'is_end', 'data', 'count']

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.data: Any = None  # Store associated data (e.g., message IDs)
        self.count: int = 0  # Frequency count


class Trie:
    """
    Trie (Prefix Tree) for fast prefix-based search and autocomplete.

    - O(k) insert/search where k is key length
    - O(p + n) prefix search where p is prefix length, n is results
    - Use case: Autocomplete usernames, find all messages starting with prefix

    Example:
        trie = Trie()
        trie.insert("@username1", message_ids=[1, 2, 3])
        trie.insert("@username2", message_ids=[4, 5])

        results = trie.search_prefix("@user")  # Returns both
        completions = trie.autocomplete("@user", limit=5)
    """

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, key: str, data: Any = None) -> None:
        """Insert a key with optional associated data. O(k)."""
        node = self.root
        for char in key.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1

        if not node.is_end:
            self.size += 1
        node.is_end = True

        # Store or append data
        if data is not None:
            if node.data is None:
                node.data = []
            if isinstance(data, list):
                node.data.extend(data)
            else:
                node.data.append(data)

    def search(self, key: str) -> Optional[Any]:
        """Search for exact key. O(k). Returns associated data or None."""
        node = self._find_node(key.lower())
        return node.data if node and node.is_end else None

    def __contains__(self, key: str) -> bool:
        """Check if key exists. O(k)."""
        node = self._find_node(key.lower())
        return node is not None and node.is_end

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node for a given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def search_prefix(self, prefix: str) -> list[tuple[str, Any]]:
        """
        Find all keys with given prefix. O(p + n).
        Returns list of (key, data) tuples.
        """
        results = []
        node = self._find_node(prefix.lower())
        if node:
            self._collect_all(node, prefix.lower(), results)
        return results

    def _collect_all(
        self,
        node: TrieNode,
        prefix: str,
        results: list[tuple[str, Any]]
    ) -> None:
        """Recursively collect all keys under a node."""
        if node.is_end:
            results.append((prefix, node.data))
        for char, child in node.children.items():
            self._collect_all(child, prefix + char, results)

    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        """
        Get autocomplete suggestions for prefix.
        Returns most frequent completions up to limit.
        """
        node = self._find_node(prefix.lower())
        if not node:
            return []

        suggestions = []
        self._collect_suggestions(node, prefix.lower(), suggestions)

        # Sort by frequency and return top results
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[:limit]]

    def _collect_suggestions(
        self,
        node: TrieNode,
        prefix: str,
        suggestions: list[tuple[str, int]]
    ) -> None:
        """Collect suggestions with their frequency counts."""
        if node.is_end:
            suggestions.append((prefix, node.count))
        for char, child in node.children.items():
            self._collect_suggestions(child, prefix + char, suggestions)

    def __len__(self) -> int:
        return self.size


# ============================================
# LRU CACHE
# ============================================

class LRUCache:
    """
    Least Recently Used (LRU) Cache for query results.

    - O(1) get/put operations
    - Automatically evicts least recently used items when full
    - Use case: Cache expensive query results

    Example:
        cache = LRUCache(maxsize=1000)
        cache.put("query:hello", results)
        results = cache.get("query:hello")  # O(1)
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache. O(1). Returns None if not found."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache. O(1). Evicts LRU item if full."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate
        }


def lru_cached(cache: LRUCache, key_func: Callable[..., str] = None):
    """
    Decorator to cache function results using LRUCache.

    Example:
        cache = LRUCache(1000)

        @lru_cached(cache, key_func=lambda q, **kw: f"search:{q}")
        def search(query, limit=100):
            return expensive_search(query, limit)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        return wrapper
    return decorator


# ============================================
# GRAPH ALGORITHMS FOR REPLY THREADS
# ============================================

class ReplyGraph:
    """
    Graph structure for message reply relationships.

    Supports:
    - DFS: Depth-first traversal for finding all descendants
    - BFS: Breadth-first traversal for level-order exploration
    - Connected components: Find isolated conversation threads
    - Topological sort: Order messages by reply chain

    Time complexity: O(V + E) for traversals
    Space complexity: O(V) for visited set
    """

    def __init__(self):
        # Adjacency lists
        self.children: dict[int, list[int]] = defaultdict(list)  # parent -> [children]
        self.parents: dict[int, int] = {}  # child -> parent
        self.nodes: set[int] = set()

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Add a reply relationship. O(1)."""
        self.children[parent_id].append(child_id)
        self.parents[child_id] = parent_id
        self.nodes.add(parent_id)
        self.nodes.add(child_id)

    def add_message(self, message_id: int, reply_to: Optional[int] = None) -> None:
        """Add a message, optionally with its reply relationship."""
        self.nodes.add(message_id)
        if reply_to is not None:
            self.add_edge(reply_to, message_id)

    def get_children(self, message_id: int) -> list[int]:
        """Get direct replies to a message. O(1)."""
        return self.children.get(message_id, [])

    def get_parent(self, message_id: int) -> Optional[int]:
        """Get the message this is a reply to. O(1)."""
        return self.parents.get(message_id)

    # ==================
    # DFS - Depth First Search
    # ==================

    def dfs_descendants(self, start_id: int) -> list[int]:
        """
        DFS: Get all descendants of a message (entire sub-thread).

        Time: O(V + E)
        Space: O(V)

        Returns messages in DFS order (deep before wide).
        """
        result = []
        visited = set()

        def dfs(node_id: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            result.append(node_id)
            for child_id in self.children.get(node_id, []):
                dfs(child_id)

        dfs(start_id)
        return result

    def dfs_iterative(self, start_id: int) -> Iterator[int]:
        """
        Iterative DFS using explicit stack (avoids recursion limit).

        Yields message IDs in DFS order.
        """
        stack = [start_id]
        visited = set()

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            yield node_id

            # Add children in reverse order for correct DFS order
            for child_id in reversed(self.children.get(node_id, [])):
                if child_id not in visited:
                    stack.append(child_id)

    # ==================
    # BFS - Breadth First Search
    # ==================

    def bfs_descendants(self, start_id: int) -> list[int]:
        """
        BFS: Get all descendants level by level.

        Time: O(V + E)
        Space: O(V)

        Returns messages in BFS order (level by level).
        """
        result = []
        visited = set()
        queue = deque([start_id])

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append(node_id)

            for child_id in self.children.get(node_id, []):
                if child_id not in visited:
                    queue.append(child_id)

        return result

    def bfs_with_depth(self, start_id: int) -> list[tuple[int, int]]:
        """
        BFS with depth information.

        Returns list of (message_id, depth) tuples.
        """
        result = []
        visited = set()
        queue = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append((node_id, depth))

            for child_id in self.children.get(node_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        return result

    # ==================
    # THREAD RECONSTRUCTION
    # ==================

    def get_thread_root(self, message_id: int) -> int:
        """
        Find the root message of a thread. O(d) where d is depth.
        """
        current = message_id
        while current in self.parents:
            current = self.parents[current]
        return current

    def get_full_thread(self, message_id: int) -> list[int]:
        """
        Get the complete thread containing a message.

        1. Find root via parent traversal
        2. BFS from root to get all descendants
        """
        root = self.get_thread_root(message_id)
        return self.bfs_descendants(root)

    def get_ancestors(self, message_id: int) -> list[int]:
        """
        Get all ancestors (path to root). O(d).

        Returns in order from message to root.
        """
        ancestors = []
        current = message_id
        while current in self.parents:
            parent = self.parents[current]
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_thread_path(self, message_id: int) -> list[int]:
        """
        Get path from root to message. O(d).
        """
        path = [message_id]
        current = message_id
        while current in self.parents:
            parent = self.parents[current]
            path.append(parent)
            current = parent
        return list(reversed(path))

    # ==================
    # CONNECTED COMPONENTS
    # ==================

    def find_connected_components(self) -> list[set[int]]:
        """
        Find all isolated conversation threads.

        Time: O(V + E)

        Returns list of sets, each set is a connected thread.
        """
        visited = set()
        components = []

        for node in self.nodes:
            if node not in visited:
                component = set()
                # Use BFS to find all connected nodes
                queue = deque([node])
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)

                    # Add parent
                    if current in self.parents:
                        parent = self.parents[current]
                        if parent not in visited:
                            queue.append(parent)

                    # Add children
                    for child in self.children.get(current, []):
                        if child not in visited:
                            queue.append(child)

                components.append(component)

        return components

    def get_thread_roots(self) -> list[int]:
        """Get all thread root messages (messages with no parent)."""
        return [node for node in self.nodes if node not in self.parents]

    # ==================
    # STATISTICS
    # ==================

    def get_thread_depth(self, root_id: int) -> int:
        """Get maximum depth of a thread from root."""
        max_depth = 0
        for _, depth in self.bfs_with_depth(root_id):
            max_depth = max(max_depth, depth)
        return max_depth

    def get_subtree_size(self, message_id: int) -> int:
        """Get number of messages in subtree including root."""
        return len(self.dfs_descendants(message_id))

    @property
    def stats(self) -> dict:
        """Get graph statistics."""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': sum(len(children) for children in self.children.values()),
            'root_messages': len(self.get_thread_roots()),
            'connected_components': len(self.find_connected_components())
        }


# ============================================
# TRIGRAM SIMILARITY
# ============================================

def generate_trigrams(text: str) -> set[str]:
    """
    Generate trigrams (3-character subsequences) for fuzzy matching.

    Example: "hello" -> {"hel", "ell", "llo"}
    """
    text = text.lower().strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def trigram_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between trigram sets.

    Returns value between 0 (no similarity) and 1 (identical).
    """
    tri1 = generate_trigrams(text1)
    tri2 = generate_trigrams(text2)

    if not tri1 or not tri2:
        return 0.0

    intersection = len(tri1 & tri2)
    union = len(tri1 | tri2)

    return intersection / union if union > 0 else 0.0


class TrigramIndex:
    """
    Inverted index of trigrams for fuzzy search.

    Time complexity:
    - Insert: O(k) where k is text length
    - Search: O(t * m) where t is trigrams in query, m is avg matches

    Example:
        index = TrigramIndex()
        index.add(1, "שלום עולם")
        index.add(2, "שלום לכולם")

        results = index.search("שלום", threshold=0.3)
    """

    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.texts: dict[int, str] = {}

    def add(self, doc_id: int, text: str) -> None:
        """Add a document to the index."""
        self.texts[doc_id] = text
        for trigram in generate_trigrams(text):
            self.index[trigram].add(doc_id)

    def search(self, query: str, threshold: float = 0.3, limit: int = 100) -> list[tuple[int, float]]:
        """
        Search for documents similar to query.

        Returns list of (doc_id, similarity) tuples, sorted by similarity.
        """
        query_trigrams = generate_trigrams(query)
        if not query_trigrams:
            return []

        # Find candidate documents
        candidates: dict[int, int] = defaultdict(int)
        for trigram in query_trigrams:
            for doc_id in self.index.get(trigram, []):
                candidates[doc_id] += 1

        # Calculate similarity for candidates
        results = []
        query_len = len(query_trigrams)

        for doc_id, match_count in candidates.items():
            doc_trigrams = generate_trigrams(self.texts[doc_id])
            doc_len = len(doc_trigrams)

            # Jaccard similarity approximation
            similarity = match_count / (query_len + doc_len - match_count)

            if similarity >= threshold:
                results.append((doc_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def __len__(self) -> int:
        return len(self.texts)


# ============================================
# INVERTED INDEX
# ============================================

class InvertedIndex:
    """
    Simple inverted index for fast word-to-document lookup.

    Time complexity:
    - Insert: O(w) where w is word count
    - Search: O(1) for single word
    - AND/OR queries: O(min(n1, n2)) for set operations
    """

    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.doc_count = 0

    def add(self, doc_id: int, text: str) -> None:
        """Add document to index."""
        words = self._tokenize(text)
        for word in words:
            self.index[word].add(doc_id)
        self.doc_count += 1

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'[\u0590-\u05FFa-zA-Z]+', text.lower())

    def search(self, word: str) -> set[int]:
        """Find all documents containing word."""
        return self.index.get(word.lower(), set())

    def search_and(self, words: list[str]) -> set[int]:
        """Find documents containing ALL words."""
        if not words:
            return set()
        result = self.search(words[0])
        for word in words[1:]:
            result &= self.search(word)
        return result

    def search_or(self, words: list[str]) -> set[int]:
        """Find documents containing ANY word."""
        result = set()
        for word in words:
            result |= self.search(word)
        return result


if __name__ == '__main__':
    # Demo
    print("=== Bloom Filter Demo ===")
    bf = BloomFilter(expected_items=1000, fp_rate=0.01)
    bf.add("message_1")
    bf.add("message_2")
    print(f"message_1 in filter: {'message_1' in bf}")
    print(f"message_999 in filter: {'message_999' in bf}")
    print(f"Memory usage: {bf.memory_usage} bytes")

    print("\n=== Trie Demo ===")
    trie = Trie()
    trie.insert("@username1", data=1)
    trie.insert("@username2", data=2)
    trie.insert("@user_test", data=3)
    print(f"Autocomplete '@user': {trie.autocomplete('@user')}")

    print("\n=== Reply Graph Demo ===")
    graph = ReplyGraph()
    graph.add_message(1)
    graph.add_message(2, reply_to=1)
    graph.add_message(3, reply_to=1)
    graph.add_message(4, reply_to=2)
    graph.add_message(5, reply_to=2)

    print(f"DFS from 1: {graph.dfs_descendants(1)}")
    print(f"BFS from 1: {graph.bfs_descendants(1)}")
    print(f"Thread path for 4: {graph.get_thread_path(4)}")
    print(f"Stats: {graph.stats}")
