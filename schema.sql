-- Telegram Chat Indexing Schema
-- SQLite with FTS5 for full-text search

-- Main messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    type TEXT DEFAULT 'message',
    date TEXT,
    date_unixtime INTEGER,
    from_name TEXT,
    from_id TEXT,
    reply_to_message_id INTEGER,
    forwarded_from TEXT,
    forwarded_from_id TEXT,
    text_plain TEXT,
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

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text_plain,
    from_name,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
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

-- Entities table (links, mentions, etc.)
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER,
    type TEXT,
    value TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Users table (extracted from messages)
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    first_seen INTEGER,
    last_seen INTEGER,
    message_count INTEGER DEFAULT 0
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date_unixtime);
CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_id);
CREATE INDEX IF NOT EXISTS idx_messages_reply ON messages(reply_to_message_id);
CREATE INDEX IF NOT EXISTS idx_messages_forwarded ON messages(forwarded_from_id);
CREATE INDEX IF NOT EXISTS idx_entities_message ON entities(message_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(value);
