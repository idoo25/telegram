"""
Chunked Embeddings Generator
Creates embeddings for conversation windows instead of single messages.
This captures context like "Where do you live?" + "Tel Aviv" in one embedding.

Supports two chunking strategies:
1. Thread-based: Groups messages by reply chains (best for Q&A)
2. Time-window: Groups nearby messages (fallback for orphan messages)
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Set
from datetime import datetime
from collections import defaultdict


def build_thread_map(messages_db: str) -> Dict[int, List[Dict]]:
    """
    Build a map of threads from reply chains.

    Returns:
        Dict mapping root_message_id -> list of messages in thread
    """
    conn = sqlite3.connect(messages_db)
    conn.row_factory = sqlite3.Row

    # Get all messages with their reply info
    messages = conn.execute("""
        SELECT id, date, date_unixtime, from_name, text_plain, reply_to_message_id
        FROM messages
        WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
        ORDER BY date_unixtime ASC
    """).fetchall()
    conn.close()

    # Build lookup maps
    msg_by_id = {m['id']: dict(m) for m in messages}
    children = defaultdict(list)  # parent_id -> [child_ids]

    for msg in messages:
        reply_to = msg['reply_to_message_id']
        if reply_to and reply_to in msg_by_id:
            children[reply_to].append(msg['id'])

    # Find root messages (messages that others reply to, but don't reply to anything)
    # OR messages that start a chain
    roots = set()
    in_thread = set()

    for msg in messages:
        msg_id = msg['id']
        reply_to = msg['reply_to_message_id']

        if reply_to and reply_to in msg_by_id:
            # This message is part of a thread
            in_thread.add(msg_id)

            # Find the root of this thread
            current = reply_to
            while current in msg_by_id:
                parent_reply = msg_by_id[current].get('reply_to_message_id')
                if parent_reply and parent_reply in msg_by_id:
                    current = parent_reply
                else:
                    break
            roots.add(current)
            in_thread.add(current)

    # Build threads from roots
    threads = {}

    def collect_thread(root_id: int) -> List[Dict]:
        """Recursively collect all messages in a thread."""
        thread = [msg_by_id[root_id]]

        # BFS to collect children
        queue = list(children[root_id])
        while queue:
            child_id = queue.pop(0)
            if child_id in msg_by_id:
                thread.append(msg_by_id[child_id])
                queue.extend(children[child_id])

        # Sort by time
        thread.sort(key=lambda x: x.get('date_unixtime', 0))
        return thread

    for root_id in roots:
        thread = collect_thread(root_id)
        if len(thread) >= 2:  # Only keep threads with at least 2 messages
            threads[root_id] = thread

    # Find orphan messages (not in any thread)
    orphans = [msg_by_id[m['id']] for m in messages if m['id'] not in in_thread]

    print(f"Found {len(threads)} threads with {sum(len(t) for t in threads.values())} messages")
    print(f"Found {len(orphans)} orphan messages")

    return threads, orphans


def create_thread_chunks(messages_db: str, max_thread_length: int = 10) -> List[Dict]:
    """
    Create chunks based on reply threads.
    Long threads are split into sub-chunks.

    Args:
        messages_db: Path to telegram.db
        max_thread_length: Max messages per chunk (splits longer threads)

    Returns:
        List of chunk dicts
    """
    threads, orphans = build_thread_map(messages_db)

    chunks = []
    chunk_id = 0

    # Process threads
    for root_id, thread_msgs in threads.items():
        # Split long threads
        for i in range(0, len(thread_msgs), max_thread_length):
            sub_thread = thread_msgs[i:i + max_thread_length]

            chunk_lines = []
            message_ids = []

            for msg in sub_thread:
                name = msg.get('from_name') or 'Unknown'
                text = msg.get('text_plain') or ''
                date = (msg.get('date') or '')[:16]

                # Mark replies
                reply_to = msg.get('reply_to_message_id')
                prefix = "↳ " if reply_to else ""

                chunk_lines.append(f"{prefix}[{name}, {date}] {text[:200]}")
                message_ids.append(msg['id'])

            chunks.append({
                'chunk_id': chunk_id,
                'type': 'thread',
                'text': "\n".join(chunk_lines),
                'message_ids': message_ids,
                'anchor_message_id': message_ids[0],  # Root of thread
                'start_date': sub_thread[0].get('date'),
                'end_date': sub_thread[-1].get('date'),
            })
            chunk_id += 1

    # Process orphans with time-window chunking
    WINDOW_SIZE = 5
    OVERLAP = 2
    STEP = WINDOW_SIZE - OVERLAP

    orphans.sort(key=lambda x: x.get('date_unixtime', 0))

    for i in range(0, max(1, len(orphans) - WINDOW_SIZE + 1), STEP):
        window = orphans[i:i + WINDOW_SIZE]
        if not window:
            continue

        chunk_lines = []
        message_ids = []

        for msg in window:
            name = msg.get('from_name') or 'Unknown'
            text = msg.get('text_plain') or ''
            date = (msg.get('date') or '')[:16]
            chunk_lines.append(f"[{name}, {date}] {text[:200]}")
            message_ids.append(msg['id'])

        chunks.append({
            'chunk_id': chunk_id,
            'type': 'window',
            'text': "\n".join(chunk_lines),
            'message_ids': message_ids,
            'anchor_message_id': message_ids[len(message_ids)//2],
            'start_date': window[0].get('date'),
            'end_date': window[-1].get('date'),
        })
        chunk_id += 1

    print(f"Created {len(chunks)} total chunks")
    thread_chunks = sum(1 for c in chunks if c['type'] == 'thread')
    window_chunks = sum(1 for c in chunks if c['type'] == 'window')
    print(f"  - {thread_chunks} thread chunks")
    print(f"  - {window_chunks} window chunks")

    return chunks


def create_chunks(messages_db: str, window_size: int = 5, overlap: int = 2) -> List[Dict]:
    """
    Create conversation chunks from messages.

    Args:
        messages_db: Path to telegram.db
        window_size: Number of messages per chunk
        overlap: How many messages overlap between chunks

    Returns:
        List of chunk dicts with text and metadata
    """
    conn = sqlite3.connect(messages_db)
    conn.row_factory = sqlite3.Row

    # Get all messages ordered by time
    messages = conn.execute("""
        SELECT id, date, from_name, text_plain, reply_to_message_id
        FROM messages
        WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
        ORDER BY date_unixtime ASC
    """).fetchall()
    conn.close()

    print(f"Creating chunks from {len(messages):,} messages...")

    chunks = []
    step = window_size - overlap

    for i in range(0, len(messages) - window_size + 1, step):
        window = messages[i:i + window_size]

        # Build chunk text with speaker context
        chunk_lines = []
        message_ids = []

        for msg in window:
            name = msg['from_name'] or 'Unknown'
            text = msg['text_plain'] or ''
            date = msg['date'] or ''

            # Format: [Name, Date] Message
            if len(date) > 16:
                date = date[:16]  # Just date and time
            chunk_lines.append(f"[{name}, {date}] {text[:200]}")
            message_ids.append(msg['id'])

        chunk_text = "\n".join(chunk_lines)

        # Use middle message as the "anchor"
        anchor_idx = window_size // 2
        anchor_msg = window[anchor_idx]

        chunks.append({
            'chunk_id': i // step,
            'text': chunk_text,
            'message_ids': message_ids,
            'anchor_message_id': anchor_msg['id'],
            'start_date': window[0]['date'],
            'end_date': window[-1]['date'],
        })

    print(f"Created {len(chunks):,} chunks")
    return chunks


def generate_chunk_embeddings(chunks: List[Dict], model_name: str = 'intfloat/multilingual-e5-large'):
    """Generate embeddings for chunks."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)

    # Prepare texts with e5 prefix
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    print(f"Generating embeddings for {len(texts):,} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    return embeddings


def save_chunk_embeddings(chunks: List[Dict], embeddings: np.ndarray, output_db: str = 'chunk_embeddings.db'):
    """Save chunk embeddings to database."""
    import json

    conn = sqlite3.connect(output_db)

    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY,
            text TEXT,
            message_ids TEXT,
            anchor_message_id INTEGER,
            start_date TEXT,
            end_date TEXT,
            embedding BLOB
        )
    """)

    # Clear existing data
    conn.execute("DELETE FROM chunk_embeddings")

    # Insert chunks
    data = []
    for chunk, emb in zip(chunks, embeddings):
        data.append((
            chunk['chunk_id'],
            chunk['text'],
            json.dumps(chunk['message_ids']),
            chunk['anchor_message_id'],
            chunk['start_date'],
            chunk['end_date'],
            emb.astype(np.float32).tobytes()
        ))

    conn.executemany(
        "INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?, ?, ?)",
        data
    )
    conn.commit()

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    conn.close()

    print(f"Saved {count:,} chunk embeddings to {output_db}")
    return count


def create_bm25_index(messages_db: str, output_db: str = 'search_index.db'):
    """Create BM25 index for hybrid search."""
    from rank_bm25 import BM25Okapi
    import pickle
    import re

    conn = sqlite3.connect(messages_db)
    messages = conn.execute("""
        SELECT id, from_name, text_plain
        FROM messages
        WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
    """).fetchall()
    conn.close()

    print(f"Creating BM25 index for {len(messages):,} messages...")

    # Tokenize (simple word split)
    def tokenize(text):
        # Simple tokenization - split on non-word chars
        return re.findall(r'\w+', text.lower())

    message_ids = [m[0] for m in messages]
    texts = [m[2] for m in messages]
    tokenized = [tokenize(t) for t in texts]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized)

    # Save index
    index_conn = sqlite3.connect(output_db)
    index_conn.execute("""
        CREATE TABLE IF NOT EXISTS bm25_index (
            id INTEGER PRIMARY KEY,
            data BLOB
        )
    """)
    index_conn.execute("DELETE FROM bm25_index")

    # Save as pickle
    index_data = {
        'bm25': bm25,
        'message_ids': message_ids,
        'tokenized': tokenized
    }
    index_conn.execute(
        "INSERT INTO bm25_index VALUES (1, ?)",
        (pickle.dumps(index_data),)
    )
    index_conn.commit()
    index_conn.close()

    print(f"BM25 index saved to {output_db}")
    return len(messages)


# ============================================
# COLAB SCRIPT (copy this to Colab)
# ============================================

COLAB_SCRIPT = '''
# === Run this in Google Colab ===

# Install dependencies
!pip install sentence-transformers huggingface_hub rank_bm25

# Download DB from HuggingFace
from huggingface_hub import hf_hub_download
import shutil

hf_token = "YOUR_HF_TOKEN"  # Replace with your token

path = hf_hub_download(
    repo_id='rottg/telegram-db',
    filename='telegram.db',
    repo_type='dataset',
    token=hf_token
)
shutil.copy(path, 'telegram.db')
print("Downloaded telegram.db")

# === Create Chunked Embeddings ===
import sqlite3
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load messages
conn = sqlite3.connect('telegram.db')
conn.row_factory = sqlite3.Row
messages = conn.execute("""
    SELECT id, date, from_name, text_plain
    FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
    ORDER BY date_unixtime ASC
""").fetchall()
conn.close()

print(f"Loaded {len(messages):,} messages")

# Create chunks (5-message windows, 2-message overlap)
WINDOW_SIZE = 5
OVERLAP = 2
STEP = WINDOW_SIZE - OVERLAP

chunks = []
for i in tqdm(range(0, len(messages) - WINDOW_SIZE + 1, STEP)):
    window = messages[i:i + WINDOW_SIZE]

    chunk_lines = []
    message_ids = []

    for msg in window:
        name = msg['from_name'] or 'Unknown'
        text = msg['text_plain'] or ''
        date = (msg['date'] or '')[:16]
        chunk_lines.append(f"[{name}, {date}] {text[:200]}")
        message_ids.append(msg['id'])

    chunks.append({
        'chunk_id': i // STEP,
        'text': "\\n".join(chunk_lines),
        'message_ids': message_ids,
        'anchor_message_id': window[WINDOW_SIZE // 2]['id'],
        'start_date': window[0]['date'],
        'end_date': window[-1]['date'],
    })

print(f"Created {len(chunks):,} chunks")

# Generate embeddings
model = SentenceTransformer('intfloat/multilingual-e5-large')
texts = [f"passage: {c['text']}" for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

print(f"Generated {len(embeddings):,} embeddings")

# Save to DB
emb_conn = sqlite3.connect('chunk_embeddings.db')
emb_conn.execute("""
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id INTEGER PRIMARY KEY,
        text TEXT,
        message_ids TEXT,
        anchor_message_id INTEGER,
        start_date TEXT,
        end_date TEXT,
        embedding BLOB
    )
""")
emb_conn.execute("DELETE FROM chunk_embeddings")

data = [(c['chunk_id'], c['text'], json.dumps(c['message_ids']),
         c['anchor_message_id'], c['start_date'], c['end_date'],
         emb.astype(np.float32).tobytes())
        for c, emb in zip(chunks, embeddings)]

emb_conn.executemany("INSERT INTO chunk_embeddings VALUES (?,?,?,?,?,?,?)", data)
emb_conn.commit()
emb_conn.close()

print("Saved chunk_embeddings.db")

# === Create BM25 Index ===
from rank_bm25 import BM25Okapi
import pickle
import re

def tokenize(text):
    return re.findall(r'\\w+', text.lower())

conn = sqlite3.connect('telegram.db')
messages = conn.execute("""
    SELECT id, text_plain FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
""").fetchall()
conn.close()

message_ids = [m[0] for m in messages]
tokenized = [tokenize(m[1]) for m in messages]
bm25 = BM25Okapi(tokenized)

# Save BM25 index
with open('bm25_index.pkl', 'wb') as f:
    pickle.dump({'bm25': bm25, 'message_ids': message_ids}, f)

print("Saved bm25_index.pkl")

# === Upload to HuggingFace ===
from huggingface_hub import HfApi
api = HfApi(token=hf_token)

api.upload_file(
    path_or_fileobj='chunk_embeddings.db',
    path_in_repo='chunk_embeddings.db',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("Uploaded chunk_embeddings.db")

api.upload_file(
    path_or_fileobj='bm25_index.pkl',
    path_in_repo='bm25_index.pkl',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("Uploaded bm25_index.pkl")

print("\\n=== DONE! ===")
'''

if __name__ == '__main__':
    print("=== Chunked Embeddings Generator ===")
    print("\nFor Colab, use this script:")
    print("-" * 50)
    print(COLAB_SCRIPT)
