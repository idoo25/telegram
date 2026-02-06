"""
Chunked Embeddings Generator
Creates embeddings for conversation windows instead of single messages.
This captures context like "Where do you live?" + "Tel Aviv" in one embedding.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


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
