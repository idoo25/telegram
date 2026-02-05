-- Telegram Chat Indexing Schema (Optimized)
-- SQLite with FTS5 for full-text search + performance optimizations

-- ============================================
-- PRAGMA OPTIMIZATIONS
-- ============================================
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging for better concurrency
PRAGMA synchronous = NORMAL;         -- Balance between safety and speed
PRAGMA cache_size = -64000;          -- 64MB cache
PRAGMA temp_store = MEMORY;          -- Store temp tables in memory
PRAGMA mmap_size = 268435456;        -- 256MB memory-mapped I/O

-- ============================================
-- MAIN TABLES
-- ============================================

-- Main messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    type TEXT DEFAULT 'message',
    date TEXT,
    date_unixtime INTEGER NOT NULL,
    from_name TEXT,
    from_id TEXT NOT NULL,
    reply_to_message_id INTEGER,
    forwarded_from TEXT,
    forwarded_from_id TEXT,
    text_plain TEXT,
    text_length INTEGER DEFAULT 0,
    has_media INTEGER DEFAULT 0,
    has_photo INTEGER DEFAULT 0,
    has_links INTEGER DEFAULT 0,
    has_mentions INTEGER DEFAULT 0,
    is_edited INTEGER DEFAULT 0,
    edited_unixtime INTEGER,
    photo_file_size INTEGER,
    photo_width INTEGER,
    photo_height INTEGER,
    raw_json TEXT
);

-- Users table (extracted from messages)
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    first_seen INTEGER,
    last_seen INTEGER,
    message_count INTEGER DEFAULT 0
);

-- Entities table (links, mentions, etc.)
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- ============================================
-- GRAPH STRUCTURE FOR REPLY THREADS
-- ============================================

-- Pre-computed reply graph edges for fast traversal
CREATE TABLE IF NOT EXISTS reply_graph (
    parent_id INTEGER NOT NULL,
    child_id INTEGER NOT NULL,
    depth INTEGER DEFAULT 1,
    PRIMARY KEY (parent_id, child_id)
);

-- Conversation threads (connected components)
CREATE TABLE IF NOT EXISTS threads (
    thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_message_id INTEGER UNIQUE,
    message_count INTEGER DEFAULT 0,
    first_message_time INTEGER,
    last_message_time INTEGER,
    participant_count INTEGER DEFAULT 0
);

-- Message to thread mapping
CREATE TABLE IF NOT EXISTS message_threads (
    message_id INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL,
    depth INTEGER DEFAULT 0,
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
);

-- ============================================
-- TRIGRAM INDEX FOR FUZZY SEARCH
-- ============================================

-- Trigrams for fuzzy/approximate string matching
CREATE TABLE IF NOT EXISTS trigrams (
    trigram TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (trigram, message_id, position)
);

-- ============================================
-- FTS5 FULL-TEXT SEARCH (OPTIMIZED)
-- ============================================

-- Full-text search with prefix index for autocomplete
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text_plain,
    from_name,
    content='messages',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3 4'  -- Enable prefix queries for autocomplete
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text_plain, from_name)
    VALUES (new.id, new.text_plain, new.from_name);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text_plain, from_name)
    VALUES ('delete', old.id, old.text_plain, old.from_name);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text_plain, from_name)
    VALUES ('delete', old.id, old.text_plain, old.from_name);
    INSERT INTO messages_fts(rowid, text_plain, from_name)
    VALUES (new.id, new.text_plain, new.from_name);
END;

-- ============================================
-- OPTIMIZED INDEXES
-- ============================================

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date_unixtime);
CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_id);
CREATE INDEX IF NOT EXISTS idx_messages_from_date ON messages(from_id, date_unixtime);
CREATE INDEX IF NOT EXISTS idx_messages_reply ON messages(reply_to_message_id) WHERE reply_to_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_forwarded ON messages(forwarded_from_id) WHERE forwarded_from_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_has_links ON messages(has_links) WHERE has_links = 1;
CREATE INDEX IF NOT EXISTS idx_messages_has_media ON messages(has_media) WHERE has_media = 1;

-- Entity indexes
CREATE INDEX IF NOT EXISTS idx_entities_message ON entities(message_id);
CREATE INDEX IF NOT EXISTS idx_entities_type_value ON entities(type, value);
CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(value);

-- Graph indexes
CREATE INDEX IF NOT EXISTS idx_reply_graph_child ON reply_graph(child_id);
CREATE INDEX IF NOT EXISTS idx_message_threads_thread ON message_threads(thread_id);

-- Trigram index
CREATE INDEX IF NOT EXISTS idx_trigrams_trigram ON trigrams(trigram);

-- ============================================
-- PARTICIPANTS TABLE (from Telethon API)
-- ============================================

CREATE TABLE IF NOT EXISTS participants (
    user_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    username TEXT,
    phone TEXT,
    is_bot INTEGER DEFAULT 0,
    is_admin INTEGER DEFAULT 0,
    is_creator INTEGER DEFAULT 0,
    is_premium INTEGER DEFAULT 0,
    join_date INTEGER,
    last_status TEXT DEFAULT 'unknown',
    last_online INTEGER,
    about TEXT,
    updated_at INTEGER
);

-- ============================================
-- STATISTICS TABLE FOR FAST AGGREGATIONS
-- ============================================

CREATE TABLE IF NOT EXISTS stats_cache (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
);

-- ============================================
-- VECTOR EMBEDDINGS TABLE (OPTIONAL)
-- ============================================

-- For semantic search with FAISS
CREATE TABLE IF NOT EXISTS embeddings (
    message_id INTEGER PRIMARY KEY,
    embedding BLOB,  -- Serialized numpy array
    model_name TEXT DEFAULT 'default',
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);
