#!/usr/bin/env python3
"""
Advanced Algorithms Module for Telegram Chat Analysis

Implements algorithms from Data Structures course:
- LCS (Longest Common Subsequence) - Similar message detection
- Heap-based Top-K - Efficient ranking without full sort
- Selection Algorithm (Median of Medians) - O(n) percentiles
- Rank Tree (Order Statistics Tree) - O(log n) rank queries
- Bucket Sort - O(n) time-based histograms

All algorithms are optimized for the chat indexing use case.
"""

import heapq
from typing import Any, Callable, Generator, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ============================================
# LCS - LONGEST COMMON SUBSEQUENCE
# ============================================

def lcs_length(s1: str, s2: str) -> int:
    """
    Calculate length of Longest Common Subsequence.

    Time: O(m * n)
    Space: O(min(m, n)) - optimized to use less space

    Use case: Measure similarity between two messages.
    """
    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Use two rows instead of full matrix
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                curr[i] = prev[i-1] + 1
            else:
                curr[i] = max(prev[i], curr[i-1])
        prev, curr = curr, prev

    return prev[m]


def lcs_string(s1: str, s2: str) -> str:
    """
    Find the actual Longest Common Subsequence string.

    Time: O(m * n)
    Space: O(m * n)

    Use case: Find common content between messages.
    """
    m, n = len(s1), len(s2)

    # Build full DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find the actual subsequence
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(result))


def lcs_similarity(s1: str, s2: str) -> float:
    """
    Calculate LCS-based similarity ratio between two strings.

    Returns value between 0 (no similarity) and 1 (identical).

    Use case: Detect near-duplicate messages, reposts.
    """
    if not s1 or not s2:
        return 0.0

    lcs_len = lcs_length(s1, s2)
    max_len = max(len(s1), len(s2))

    return lcs_len / max_len


def find_similar_messages(
    messages: list[tuple[int, str]],
    threshold: float = 0.7,
    min_length: int = 20
) -> list[tuple[int, int, float]]:
    """
    Find pairs of similar messages using LCS.

    Args:
        messages: List of (id, text) tuples
        threshold: Minimum similarity to report (0-1)
        min_length: Minimum message length to consider

    Returns:
        List of (id1, id2, similarity) tuples

    Time: O(n² * m) where n=messages, m=avg length
    """
    # Filter by length
    filtered = [(id_, text) for id_, text in messages if len(text) >= min_length]

    similar_pairs = []
    n = len(filtered)

    for i in range(n):
        for j in range(i + 1, n):
            id1, text1 = filtered[i]
            id2, text2 = filtered[j]

            # Quick length check - if lengths differ too much, skip
            len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
            if len_ratio < threshold:
                continue

            sim = lcs_similarity(text1, text2)
            if sim >= threshold:
                similar_pairs.append((id1, id2, sim))

    return sorted(similar_pairs, key=lambda x: x[2], reverse=True)


# ============================================
# HEAP-BASED TOP-K
# ============================================

class TopK:
    """
    Efficient Top-K tracker using min-heap.

    Maintains the K largest elements seen so far.

    Time: O(n log k) for n insertions
    Space: O(k)

    Use case: Top users, top words, top domains without sorting all data.
    """

    def __init__(self, k: int, key: Callable[[Any], float] = None):
        """
        Args:
            k: Number of top elements to track
            key: Function to extract comparison value (default: identity)
        """
        self.k = k
        self.key = key or (lambda x: x)
        self.heap: list[tuple[float, int, Any]] = []  # (key_value, counter, item)
        self.counter = 0  # For stable sorting

    def push(self, item: Any) -> None:
        """Add an item. O(log k)."""
        key_val = self.key(item)

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (key_val, self.counter, item))
        elif key_val > self.heap[0][0]:
            heapq.heapreplace(self.heap, (key_val, self.counter, item))

        self.counter += 1

    def get_top(self) -> list[Any]:
        """Get top K items sorted by key descending. O(k log k)."""
        return [item for _, _, item in sorted(self.heap, reverse=True)]

    def __len__(self) -> int:
        return len(self.heap)


def top_k_frequent(items: list[Any], k: int) -> list[tuple[Any, int]]:
    """
    Find top K most frequent items.

    Time: O(n + m log k) where n=items, m=unique items
    Space: O(m)

    Use case: Top words, top users, top mentioned usernames.
    """
    # Count frequencies
    freq = defaultdict(int)
    for item in items:
        freq[item] += 1

    # Use heap to find top K
    top = TopK(k, key=lambda x: x[1])
    for item, count in freq.items():
        top.push((item, count))

    return top.get_top()


def top_k_by_field(
    records: list[dict],
    field: str,
    k: int,
    reverse: bool = True
) -> list[dict]:
    """
    Get top K records by a specific field value.

    Time: O(n log k)

    Use case: Top messages by length, top users by message count.
    """
    if reverse:
        # Max K - use min heap
        top = TopK(k, key=lambda x: x.get(field, 0))
    else:
        # Min K - negate the key
        top = TopK(k, key=lambda x: -x.get(field, 0))

    for record in records:
        top.push(record)

    return top.get_top()


# ============================================
# SELECTION ALGORITHM (MEDIAN OF MEDIANS)
# ============================================

def partition(arr: list, left: int, right: int, pivot_idx: int) -> int:
    """
    Partition array around pivot (Lomuto scheme).

    Returns final position of pivot.
    """
    pivot_val = arr[pivot_idx]

    # Move pivot to end
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    store_idx = left
    for i in range(left, right):
        if arr[i] < pivot_val:
            arr[store_idx], arr[i] = arr[i], arr[store_idx]
            store_idx += 1

    # Move pivot to final position
    arr[store_idx], arr[right] = arr[right], arr[store_idx]

    return store_idx


def median_of_five(arr: list, left: int, right: int) -> int:
    """Find median of up to 5 elements, return its index."""
    sub = [(arr[i], i) for i in range(left, right + 1)]
    sub.sort()
    return sub[len(sub) // 2][1]


def median_of_medians(arr: list, left: int, right: int) -> int:
    """
    Find approximate median using median-of-medians algorithm.

    Returns index of the pivot.
    """
    n = right - left + 1

    if n <= 5:
        return median_of_five(arr, left, right)

    # Divide into groups of 5 and find medians
    medians = []
    for i in range(left, right + 1, 5):
        group_right = min(i + 4, right)
        median_idx = median_of_five(arr, i, group_right)
        medians.append(arr[median_idx])

    # Recursively find median of medians
    # For simplicity, use sorting for small arrays
    medians.sort()
    pivot_val = medians[len(medians) // 2]

    # Find index of this value in original array
    for i in range(left, right + 1):
        if arr[i] == pivot_val:
            return i

    return left  # Fallback


def quickselect(arr: list, k: int) -> Any:
    """
    Find the k-th smallest element (0-indexed).

    Time: O(n) average, O(n) worst case with median-of-medians
    Space: O(1) - in-place

    Use case: Find median, percentiles without sorting.
    """
    arr = arr.copy()  # Don't modify original
    left, right = 0, len(arr) - 1

    while left < right:
        # Use median of medians for pivot selection
        pivot_idx = median_of_medians(arr, left, right)
        pivot_idx = partition(arr, left, right, pivot_idx)

        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            right = pivot_idx - 1
        else:
            left = pivot_idx + 1

    return arr[left]


def find_median(arr: list) -> float:
    """
    Find median in O(n) time.

    Use case: Median message length, median activity time.
    """
    n = len(arr)
    if n == 0:
        return 0.0

    if n % 2 == 1:
        return float(quickselect(arr, n // 2))
    else:
        return (quickselect(arr, n // 2 - 1) + quickselect(arr, n // 2)) / 2


def find_percentile(arr: list, p: float) -> float:
    """
    Find the p-th percentile (0-100) in O(n) time.

    Use case: 90th percentile response time, activity distribution.
    """
    if not arr:
        return 0.0

    k = int((p / 100) * (len(arr) - 1))
    return float(quickselect(arr, k))


# ============================================
# RANK TREE (ORDER STATISTICS TREE)
# ============================================

@dataclass
class RankTreeNode:
    """Node in an Order Statistics Tree (augmented BST)."""
    key: Any
    value: Any = None
    left: 'RankTreeNode' = None
    right: 'RankTreeNode' = None
    size: int = 1  # Size of subtree (for rank queries)
    height: int = 1  # For AVL balancing


class RankTree:
    """
    Order Statistics Tree with AVL balancing.

    Supports:
    - O(log n) insert, delete, search
    - O(log n) select(k) - find k-th smallest
    - O(log n) rank(x) - find rank of element x

    Use case: "What rank is this user?", "Who is the 100th most active?"
    """

    def __init__(self, key_func: Callable[[Any], Any] = None):
        self.root: Optional[RankTreeNode] = None
        self.key_func = key_func or (lambda x: x)

    def _get_size(self, node: RankTreeNode) -> int:
        return node.size if node else 0

    def _get_height(self, node: RankTreeNode) -> int:
        return node.height if node else 0

    def _get_balance(self, node: RankTreeNode) -> int:
        return self._get_height(node.left) - self._get_height(node.right) if node else 0

    def _update(self, node: RankTreeNode) -> None:
        """Update size and height of a node."""
        if node:
            node.size = 1 + self._get_size(node.left) + self._get_size(node.right)
            node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _rotate_right(self, y: RankTreeNode) -> RankTreeNode:
        """Right rotation for AVL balance."""
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self._update(y)
        self._update(x)

        return x

    def _rotate_left(self, x: RankTreeNode) -> RankTreeNode:
        """Left rotation for AVL balance."""
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self._update(x)
        self._update(y)

        return y

    def _balance(self, node: RankTreeNode) -> RankTreeNode:
        """Balance the node if needed (AVL)."""
        self._update(node)
        balance = self._get_balance(node)

        # Left heavy
        if balance > 1:
            if self._get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right heavy
        if balance < -1:
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def insert(self, key: Any, value: Any = None) -> None:
        """Insert a key-value pair. O(log n)."""
        self.root = self._insert(self.root, key, value)

    def _insert(self, node: RankTreeNode, key: Any, value: Any) -> RankTreeNode:
        if not node:
            return RankTreeNode(key=key, value=value)

        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value  # Update existing
            return node

        return self._balance(node)

    def select(self, k: int) -> Optional[Any]:
        """
        Find the k-th smallest element (1-indexed).

        O(log n)

        Use case: "Who is the 10th most active user?"
        """
        return self._select(self.root, k)

    def _select(self, node: RankTreeNode, k: int) -> Optional[Any]:
        if not node:
            return None

        left_size = self._get_size(node.left)

        if k == left_size + 1:
            return node.value
        elif k <= left_size:
            return self._select(node.left, k)
        else:
            return self._select(node.right, k - left_size - 1)

    def rank(self, key: Any) -> int:
        """
        Find the rank of an element (1-indexed).

        O(log n)

        Use case: "What rank is user X?"
        """
        return self._rank(self.root, key)

    def _rank(self, node: RankTreeNode, key: Any) -> int:
        if not node:
            return 0

        if key < node.key:
            return self._rank(node.left, key)
        elif key > node.key:
            return 1 + self._get_size(node.left) + self._rank(node.right, key)
        else:
            return self._get_size(node.left) + 1

    def __len__(self) -> int:
        return self._get_size(self.root)

    def inorder(self) -> Generator[tuple[Any, Any], None, None]:
        """Iterate in sorted order."""
        def _inorder(node):
            if node:
                yield from _inorder(node.left)
                yield (node.key, node.value)
                yield from _inorder(node.right)
        yield from _inorder(self.root)


# ============================================
# BUCKET SORT FOR TIME-BASED DATA
# ============================================

def bucket_sort_by_time(
    records: list[dict],
    time_field: str,
    bucket_size: int = 3600,  # Default: 1 hour
    start_time: int = None,
    end_time: int = None
) -> list[list[dict]]:
    """
    Sort records into time-based buckets.

    Time: O(n + k) where k = number of buckets
    Space: O(n)

    Use case: Group messages by hour, day, week for histograms.

    Args:
        records: List of dicts with timestamp field
        time_field: Name of the timestamp field
        bucket_size: Size of each bucket in seconds
        start_time: Start of range (default: min timestamp)
        end_time: End of range (default: max timestamp)

    Returns:
        List of buckets, each containing records in that time range
    """
    if not records:
        return []

    # Extract timestamps
    timestamps = [r.get(time_field, 0) for r in records]

    if start_time is None:
        start_time = min(timestamps)
    if end_time is None:
        end_time = max(timestamps)

    # Calculate number of buckets
    n_buckets = max(1, (end_time - start_time) // bucket_size + 1)

    # Initialize buckets
    buckets: list[list[dict]] = [[] for _ in range(n_buckets)]

    # Distribute records into buckets
    for record in records:
        ts = record.get(time_field, 0)
        if ts < start_time or ts > end_time:
            continue

        bucket_idx = min((ts - start_time) // bucket_size, n_buckets - 1)
        buckets[bucket_idx].append(record)

    return buckets


def time_histogram(
    records: list[dict],
    time_field: str,
    bucket_size: int = 3600
) -> list[tuple[int, int]]:
    """
    Create a histogram of record counts over time.

    Returns list of (bucket_start_time, count) tuples.

    Use case: Activity over time visualization.
    """
    if not records:
        return []

    timestamps = [r.get(time_field, 0) for r in records]
    start_time = min(timestamps)
    end_time = max(timestamps)

    buckets = bucket_sort_by_time(records, time_field, bucket_size, start_time, end_time)

    result = []
    for i, bucket in enumerate(buckets):
        bucket_time = start_time + i * bucket_size
        result.append((bucket_time, len(bucket)))

    return result


def hourly_distribution(
    records: list[dict],
    time_field: str
) -> dict[int, int]:
    """
    Get distribution of records by hour of day (0-23).

    Time: O(n)

    Use case: When are users most active?
    """
    from datetime import datetime

    dist = defaultdict(int)

    for record in records:
        ts = record.get(time_field, 0)
        if ts:
            hour = datetime.fromtimestamp(ts).hour
            dist[hour] += 1

    return dict(dist)


# ============================================
# COMBINED DATA STRUCTURE: RANKED TIME INDEX
# ============================================

class RankedTimeIndex:
    """
    Combined data structure for efficient time-based and rank queries.

    Combines:
    - Bucket sort for O(1) time range access
    - Rank tree for O(log n) rank queries
    - Top-K heap for efficient top queries

    Use case: "Top 10 users in the last hour", "Rank of user X this week"
    """

    def __init__(self, bucket_size: int = 3600):
        self.bucket_size = bucket_size
        self.buckets: dict[int, list[dict]] = defaultdict(list)  # bucket_id -> records
        self.rank_tree = RankTree()  # For rank queries
        self.total_count = 0
        self.min_time = float('inf')
        self.max_time = 0

    def add(self, record: dict, time_field: str = 'date_unixtime', rank_field: str = None) -> None:
        """Add a record to the index. O(log n)."""
        ts = record.get(time_field, 0)

        # Update time bounds
        self.min_time = min(self.min_time, ts)
        self.max_time = max(self.max_time, ts)

        # Add to time bucket
        bucket_id = ts // self.bucket_size
        self.buckets[bucket_id].append(record)

        # Add to rank tree if rank field specified
        if rank_field and rank_field in record:
            self.rank_tree.insert(record[rank_field], record)

        self.total_count += 1

    def get_time_range(self, start_time: int, end_time: int) -> list[dict]:
        """
        Get all records in time range. O(k) where k = records in range.
        """
        start_bucket = start_time // self.bucket_size
        end_bucket = end_time // self.bucket_size

        results = []
        for bucket_id in range(start_bucket, end_bucket + 1):
            for record in self.buckets.get(bucket_id, []):
                ts = record.get('date_unixtime', 0)
                if start_time <= ts <= end_time:
                    results.append(record)

        return results

    def top_k_in_range(
        self,
        start_time: int,
        end_time: int,
        k: int,
        score_field: str
    ) -> list[dict]:
        """
        Get top K records by score in time range.

        O(m log k) where m = records in range
        """
        records = self.get_time_range(start_time, end_time)
        return top_k_by_field(records, score_field, k)

    def get_rank(self, key: Any) -> int:
        """Get rank of element. O(log n)."""
        return self.rank_tree.rank(key)

    def get_by_rank(self, k: int) -> Optional[dict]:
        """Get element by rank. O(log n)."""
        return self.rank_tree.select(k)


# ============================================
# TESTS AND DEMOS
# ============================================

def run_tests():
    """Run tests for all algorithms."""
    print("=" * 60)
    print("ALGORITHM TESTS")
    print("=" * 60)

    # Test LCS
    print("\n--- LCS (Longest Common Subsequence) ---")
    s1 = "שלום לכולם מה קורה"
    s2 = "שלום לכולם מה נשמע"
    lcs = lcs_string(s1, s2)
    sim = lcs_similarity(s1, s2)
    print(f"String 1: {s1}")
    print(f"String 2: {s2}")
    print(f"LCS: '{lcs}'")
    print(f"Similarity: {sim:.2%}")

    # Test similar message detection
    messages = [
        (1, "היי מה קורה איך אתה"),
        (2, "היי מה קורה איך את"),
        (3, "שלום לכולם"),
        (4, "היי מה קורה איך אתם"),
    ]
    similar = find_similar_messages(messages, threshold=0.7, min_length=5)
    print(f"\nSimilar message pairs (threshold 0.7):")
    for id1, id2, sim in similar:
        print(f"  Messages {id1} & {id2}: {sim:.2%}")

    # Test Top-K
    print("\n--- Heap-based Top-K ---")
    items = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'date', 'banana']
    top = top_k_frequent(items, k=2)
    print(f"Items: {items}")
    print(f"Top 2 frequent: {top}")

    # Test Selection (Median)
    print("\n--- Selection Algorithm (Median) ---")
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    median = find_median(arr)
    p90 = find_percentile(arr, 90)
    print(f"Array: {arr}")
    print(f"Median: {median}")
    print(f"90th percentile: {p90}")

    # Test Rank Tree
    print("\n--- Rank Tree (Order Statistics) ---")
    tree = RankTree()
    users = [
        (100, "Alice"),
        (250, "Bob"),
        (50, "Charlie"),
        (300, "Diana"),
        (150, "Eve"),
    ]
    for score, name in users:
        tree.insert(score, name)

    print(f"Users by score: {users}")
    print(f"3rd ranked (by score): {tree.select(3)}")
    print(f"Rank of score 150: {tree.rank(150)}")
    print(f"All in order: {list(tree.inorder())}")

    # Test Bucket Sort
    print("\n--- Bucket Sort (Time-based) ---")
    records = [
        {'id': 1, 'ts': 1000},
        {'id': 2, 'ts': 1500},
        {'id': 3, 'ts': 2500},
        {'id': 4, 'ts': 1200},
        {'id': 5, 'ts': 3000},
    ]
    hist = time_histogram(records, 'ts', bucket_size=1000)
    print(f"Records: {records}")
    print(f"Histogram (bucket=1000): {hist}")

    # Test Combined Structure
    print("\n--- Combined RankedTimeIndex ---")
    index = RankedTimeIndex(bucket_size=1000)
    for r in records:
        index.add(r, time_field='ts', rank_field='id')

    range_result = index.get_time_range(1000, 2000)
    print(f"Records in time range 1000-2000: {[r['id'] for r in range_result]}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
