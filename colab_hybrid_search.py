"""
=== COLAB SCRIPT FOR HYBRID SEARCH ===
Copy and paste this into Google Colab

This creates:
1. chunk_embeddings.db - Conversation-aware embeddings (5 messages per chunk)
2. bm25_index.pkl - BM25 keyword search index

Run time: ~15-20 minutes on GPU
"""

# ============================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================
# !pip install sentence-transformers huggingface_hub rank_bm25

# ============================================
# STEP 2: DOWNLOAD DATABASE
# ============================================
"""
from huggingface_hub import hf_hub_download
import shutil

# PUT YOUR TOKEN HERE
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"

path = hf_hub_download(
    repo_id='rottg/telegram-db',
    filename='telegram.db',
    repo_type='dataset',
    token=HF_TOKEN
)
shutil.copy(path, 'telegram.db')
print("✓ Downloaded telegram.db")
"""

# ============================================
# STEP 3: CREATE CHUNK EMBEDDINGS
# ============================================
"""
import sqlite3
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load messages
print("Loading messages...")
conn = sqlite3.connect('telegram.db')
conn.row_factory = sqlite3.Row
messages = conn.execute('''
    SELECT id, date, from_name, text_plain
    FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
    ORDER BY date_unixtime ASC
''').fetchall()
conn.close()
print(f"✓ Loaded {len(messages):,} messages")

# Create chunks (5-message windows, 3 overlap)
print("Creating chunks...")
WINDOW_SIZE = 5
OVERLAP = 3
STEP = WINDOW_SIZE - OVERLAP

chunks = []
for i in tqdm(range(0, len(messages) - WINDOW_SIZE + 1, STEP), desc="Chunking"):
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

print(f"✓ Created {len(chunks):,} chunks")

# Load model and generate embeddings
print("Loading model...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Generating embeddings (this takes ~10 min)...")
texts = [f"passage: {c['text']}" for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
print(f"✓ Generated {len(embeddings):,} embeddings")

# Save to DB
print("Saving to chunk_embeddings.db...")
emb_conn = sqlite3.connect('chunk_embeddings.db')
emb_conn.execute('''
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id INTEGER PRIMARY KEY,
        text TEXT,
        message_ids TEXT,
        anchor_message_id INTEGER,
        start_date TEXT,
        end_date TEXT,
        embedding BLOB
    )
''')
emb_conn.execute("DELETE FROM chunk_embeddings")

data = [(c['chunk_id'], c['text'], json.dumps(c['message_ids']),
         c['anchor_message_id'], c['start_date'], c['end_date'],
         emb.astype(np.float32).tobytes())
        for c, emb in zip(chunks, embeddings)]

emb_conn.executemany("INSERT INTO chunk_embeddings VALUES (?,?,?,?,?,?,?)", data)
emb_conn.commit()
emb_conn.close()
print("✓ Saved chunk_embeddings.db")
"""

# ============================================
# STEP 4: CREATE BM25 INDEX
# ============================================
"""
from rank_bm25 import BM25Okapi
import pickle
import re

print("Creating BM25 index...")

conn = sqlite3.connect('telegram.db')
messages = conn.execute('''
    SELECT id, text_plain FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
''').fetchall()
conn.close()

def tokenize(text):
    return re.findall(r'\\w+', text.lower())

message_ids = [m[0] for m in messages]
tokenized = [tokenize(m[1]) for m in tqdm(messages, desc="Tokenizing")]
bm25 = BM25Okapi(tokenized)

with open('bm25_index.pkl', 'wb') as f:
    pickle.dump({'bm25': bm25, 'message_ids': message_ids}, f)

print(f"✓ Saved BM25 index ({len(message_ids):,} documents)")
"""

# ============================================
# STEP 5: UPLOAD TO HUGGINGFACE
# ============================================
"""
from huggingface_hub import HfApi
import os

api = HfApi(token=HF_TOKEN)

# Upload chunk embeddings
size_mb = os.path.getsize('chunk_embeddings.db') / (1024*1024)
print(f"Uploading chunk_embeddings.db ({size_mb:.0f} MB)...")
api.upload_file(
    path_or_fileobj='chunk_embeddings.db',
    path_in_repo='chunk_embeddings.db',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("✓ Uploaded chunk_embeddings.db")

# Upload BM25 index
size_mb = os.path.getsize('bm25_index.pkl') / (1024*1024)
print(f"Uploading bm25_index.pkl ({size_mb:.0f} MB)...")
api.upload_file(
    path_or_fileobj='bm25_index.pkl',
    path_in_repo='bm25_index.pkl',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("✓ Uploaded bm25_index.pkl")

print("\\n" + "="*50)
print("✓ DONE! Hybrid search indexes created and uploaded")
print("="*50)
"""

# ============================================
# FULL SCRIPT (uncomment and run in Colab)
# ============================================
FULL_SCRIPT = '''
# === HYBRID SEARCH WITH THREAD CHUNKING ===
# Run this entire cell in Google Colab
# Creates chunks based on reply threads (Q&A together) + time windows for orphans

!pip install -q sentence-transformers huggingface_hub rank_bm25

from huggingface_hub import hf_hub_download, HfApi
import sqlite3
import numpy as np
import json
import pickle
import re
import os
import shutil
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# === CONFIGURATION ===
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your HuggingFace token
MAX_THREAD_LENGTH = 10  # Max messages per thread chunk
WINDOW_SIZE = 5  # For orphan messages
OVERLAP = 2

# === DOWNLOAD DB ===
print("Downloading telegram.db...")
path = hf_hub_download(
    repo_id='rottg/telegram-db',
    filename='telegram.db',
    repo_type='dataset',
    token=HF_TOKEN
)
shutil.copy(path, 'telegram.db')
print("✓ Downloaded telegram.db")

# === LOAD MESSAGES ===
print("Loading messages...")
conn = sqlite3.connect('telegram.db')
conn.row_factory = sqlite3.Row
messages = conn.execute("""
    SELECT id, date, date_unixtime, from_name, text_plain, reply_to_message_id
    FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
    ORDER BY date_unixtime ASC
""").fetchall()
conn.close()
print(f"✓ Loaded {len(messages):,} messages")

# === BUILD THREAD MAP ===
print("Building thread map...")
msg_by_id = {m['id']: dict(m) for m in messages}
children = defaultdict(list)

for msg in messages:
    reply_to = msg['reply_to_message_id']
    if reply_to and reply_to in msg_by_id:
        children[reply_to].append(msg['id'])

# Find roots and messages in threads
roots = set()
in_thread = set()

for msg in messages:
    msg_id = msg['id']
    reply_to = msg['reply_to_message_id']

    if reply_to and reply_to in msg_by_id:
        in_thread.add(msg_id)
        # Find root
        current = reply_to
        while current in msg_by_id:
            parent = msg_by_id[current].get('reply_to_message_id')
            if parent and parent in msg_by_id:
                current = parent
            else:
                break
        roots.add(current)
        in_thread.add(current)

print(f"✓ Found {len(roots):,} thread roots")

# === BUILD THREADS ===
def collect_thread(root_id):
    thread = [msg_by_id[root_id]]
    queue = list(children[root_id])
    while queue:
        child_id = queue.pop(0)
        if child_id in msg_by_id:
            thread.append(msg_by_id[child_id])
            queue.extend(children[child_id])
    thread.sort(key=lambda x: x.get('date_unixtime', 0))
    return thread

threads = {}
for root_id in tqdm(roots, desc="Building threads"):
    thread = collect_thread(root_id)
    if len(thread) >= 2:
        threads[root_id] = thread

orphans = [msg_by_id[m['id']] for m in messages if m['id'] not in in_thread]
print(f"✓ {len(threads):,} threads, {len(orphans):,} orphan messages")

# === CREATE CHUNKS ===
print("Creating chunks...")
chunks = []
chunk_id = 0

# Thread chunks
for root_id, thread_msgs in tqdm(threads.items(), desc="Thread chunks"):
    for i in range(0, len(thread_msgs), MAX_THREAD_LENGTH):
        sub = thread_msgs[i:i + MAX_THREAD_LENGTH]
        lines = []
        ids = []
        for msg in sub:
            prefix = "↳ " if msg.get('reply_to_message_id') else ""
            lines.append(f"{prefix}[{msg.get('from_name', 'Unknown')}, {(msg.get('date') or '')[:16]}] {msg.get('text_plain', '')[:200]}")
            ids.append(msg['id'])
        chunks.append({
            'chunk_id': chunk_id, 'type': 'thread', 'text': "\\n".join(lines),
            'message_ids': ids, 'anchor_message_id': ids[0],
            'start_date': sub[0].get('date'), 'end_date': sub[-1].get('date')
        })
        chunk_id += 1

# Window chunks for orphans
orphans.sort(key=lambda x: x.get('date_unixtime', 0))
STEP = WINDOW_SIZE - OVERLAP
for i in tqdm(range(0, max(1, len(orphans) - WINDOW_SIZE + 1), STEP), desc="Window chunks"):
    window = orphans[i:i + WINDOW_SIZE]
    if not window: continue
    lines = []
    ids = []
    for msg in window:
        lines.append(f"[{msg.get('from_name', 'Unknown')}, {(msg.get('date') or '')[:16]}] {msg.get('text_plain', '')[:200]}")
        ids.append(msg['id'])
    chunks.append({
        'chunk_id': chunk_id, 'type': 'window', 'text': "\\n".join(lines),
        'message_ids': ids, 'anchor_message_id': ids[len(ids)//2],
        'start_date': window[0].get('date'), 'end_date': window[-1].get('date')
    })
    chunk_id += 1

thread_count = sum(1 for c in chunks if c['type'] == 'thread')
window_count = sum(1 for c in chunks if c['type'] == 'window')
print(f"✓ Created {len(chunks):,} chunks ({thread_count:,} threads, {window_count:,} windows)")

# === GENERATE EMBEDDINGS ===
print("Loading model (intfloat/multilingual-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Generating chunk embeddings...")
texts = [f"passage: {c['text']}" for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
print(f"✓ Generated {len(embeddings):,} embeddings")

# === SAVE CHUNK EMBEDDINGS ===
print("Saving chunk_embeddings.db...")
emb_conn = sqlite3.connect('chunk_embeddings.db')
emb_conn.execute("""
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id INTEGER PRIMARY KEY,
        chunk_type TEXT,
        text TEXT,
        message_ids TEXT,
        anchor_message_id INTEGER,
        start_date TEXT,
        end_date TEXT,
        embedding BLOB
    )
""")
emb_conn.execute("DELETE FROM chunk_embeddings")

data = [(c['chunk_id'], c['type'], c['text'], json.dumps(c['message_ids']),
         c['anchor_message_id'], c['start_date'], c['end_date'],
         emb.astype(np.float32).tobytes())
        for c, emb in zip(chunks, embeddings)]

emb_conn.executemany("INSERT INTO chunk_embeddings VALUES (?,?,?,?,?,?,?,?)", data)
emb_conn.commit()
emb_conn.close()
print("✓ Saved chunk_embeddings.db")

# === CREATE BM25 INDEX ===
print("Creating BM25 index...")

def tokenize(text):
    return re.findall(r'\\w+', text.lower())

conn = sqlite3.connect('telegram.db')
all_messages = conn.execute("""
    SELECT id, text_plain FROM messages
    WHERE text_plain IS NOT NULL AND LENGTH(text_plain) > 5
""").fetchall()
conn.close()

message_ids = [m[0] for m in all_messages]
tokenized = [tokenize(m[1]) for m in tqdm(all_messages, desc="Tokenizing")]
bm25 = BM25Okapi(tokenized)

with open('bm25_index.pkl', 'wb') as f:
    pickle.dump({'bm25': bm25, 'message_ids': message_ids}, f)

print(f"✓ Created BM25 index ({len(message_ids):,} documents)")

# === UPLOAD TO HUGGINGFACE ===
print("Uploading to HuggingFace...")
api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj='chunk_embeddings.db',
    path_in_repo='chunk_embeddings.db',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("✓ Uploaded chunk_embeddings.db")

api.upload_file(
    path_or_fileobj='bm25_index.pkl',
    path_in_repo='bm25_index.pkl',
    repo_id='rottg/telegram-db',
    repo_type='dataset'
)
print("✓ Uploaded bm25_index.pkl")

print()
print("=" * 60)
print("✓ HYBRID SEARCH SETUP COMPLETE!")
print("=" * 60)
print(f"  Chunks created: {len(chunks):,}")
print(f"  BM25 documents: {len(message_ids):,}")
print("  Files uploaded to HuggingFace:")
print("    - chunk_embeddings.db")
print("    - bm25_index.pkl")
print("=" * 60)
'''

if __name__ == '__main__':
    print("=" * 60)
    print("COLAB SCRIPT FOR HYBRID SEARCH")
    print("=" * 60)
    print()
    print("Copy the script below and paste into Google Colab:")
    print()
    print("-" * 60)
    print(FULL_SCRIPT)
    print("-" * 60)
