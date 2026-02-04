# Telegram JSON Indexer & Analyzer

A high-performance system for indexing, searching, and analyzing Telegram chat exports using SQLite FTS5 and advanced algorithms from Data Structures course. Includes a full-featured **Web Dashboard** with **AI-powered search**.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TELEGRAM CHAT ANALYZER                                ║
║                                                                               ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────────────┐    ║
║  │  JSON   │───▶│ INDEXER │───▶│ SQLite  │───▶│     WEB DASHBOARD       │    ║
║  │ Export  │    │ Bloom   │    │ + FTS5  │    │  ┌─────┬─────┬─────┐   │    ║
║  │         │    │ Filter  │    │         │    │  │Stats│Users│Chat │   │    ║
║  └─────────┘    └─────────┘    └─────────┘    │  ├─────┼─────┼─────┤   │    ║
║                                               │  │Search│ AI  │Mod  │   │    ║
║                                               │  └─────┴─────┴─────┘   │    ║
║                                               └─────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Features

### Core Features
- **Full-Text Search** - Fast search with Hebrew support using SQLite FTS5
- **Fuzzy Search** - Find messages even with typos using trigram similarity
- **Similar Message Detection** - LCS algorithm finds duplicates/reposts
- **Conversation Threads** - DFS/BFS traversal reconstructs reply chains
- **User Rankings** - O(log n) rank queries using AVL Rank Tree
- **Time Analytics** - Bucket Sort for efficient histograms
- **Top-K Queries** - Heap-based O(n log k) instead of O(n log n)
- **Percentiles** - O(n) median/percentiles using Selection algorithm

### Web Dashboard
- **Interactive Overview** - Charts, stats, activity graphs
- **User Leaderboard** - Rankings with detailed user profiles
- **Telegram-like Chat View** - Browse all messages like in Telegram
- **Advanced Search** - Full-text + fuzzy search with filters
- **AI-Powered Search** - Natural language queries (Hebrew/English)
- **Moderation Analytics** - Links, mentions, domains analysis
- **Database Updates** - Upload new JSON files via web UI

### AI Search (Free Providers)
- **Ollama** - Local LLM (recommended, 100% free)
- **Groq** - Free API tier available
- **Google Gemini** - Free API tier available

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Web Dashboard](#web-dashboard)
4. [AI Search](#ai-search)
5. [Database Updates](#database-updates)
6. [Architecture](#architecture)
7. [Usage Guide](#usage-guide)
8. [Algorithms](#algorithms)
9. [API Reference](#api-reference)
10. [Examples](#examples)

---

## Installation

### Requirements

- Python 3.10 or higher
- No external packages required for core functionality

### Setup

```bash
# Clone or download the project
cd telegram

# Verify Python version
python --version  # Should be 3.10+

# Test the system
python algorithms.py  # Should print "ALL TESTS PASSED!"
```

### Optional: Semantic Search

For AI-powered semantic similarity search:

```bash
pip install numpy faiss-cpu sentence-transformers
```

---

## Quick Start

### Step 1: Export from Telegram

1. Open Telegram Desktop
2. Go to any chat/group
3. Click ⋮ → Export Chat History
4. Select JSON format
5. Save as `result.json`

### Step 2: Index Your Data

```bash
python indexer.py result.json --db telegram.db
```

### Step 3: Launch Web Dashboard

```bash
# Start the dashboard (recommended)
python dashboard.py

# Open in browser: http://localhost:5000
```

### Step 4: Search & Analyze (CLI)

```bash
# Search messages
python search.py "שלום"

# View statistics
python analyzer.py --stats

# Find similar messages
python analyzer.py --similar
```

---

## Web Dashboard

The web dashboard provides a complete visual interface for analyzing your Telegram data.

### Starting the Dashboard

```bash
python dashboard.py
# Or with custom port:
python dashboard.py --port 8080
```

### Dashboard Pages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WEB DASHBOARD                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📈 Overview      │  Main statistics, charts, activity graphs           │
│                   │  - Total messages, users, links, media              │
│                   │  - Daily/hourly activity charts                     │
│                   │  - Top users leaderboard                            │
│                                                                          │
│  👥 Users         │  User leaderboard with detailed profiles            │
│                   │  - Ranking by message count                         │
│                   │  - User details modal (hourly activity)             │
│                   │  - Export users to CSV                              │
│                                                                          │
│  💬 Chat          │  Telegram-like message view                         │
│                   │  - Browse all messages chronologically              │
│                   │  - Filter by user, date, media type                 │
│                   │  - Click message to view full thread                │
│                   │  - AI search with natural language                  │
│                                                                          │
│  🔍 Search        │  Advanced search interface                          │
│                   │  - Full-text search (Hebrew supported)              │
│                   │  - AI-powered natural language search               │
│                   │  - Boolean operators (AND, OR, NOT)                 │
│                   │  - Export search results                            │
│                                                                          │
│  🛡️ Moderation    │  Content analytics                                  │
│                   │  - Top shared domains                               │
│                   │  - Most mentioned users                             │
│                   │  - Link sharers leaderboard                         │
│                   │  - Word frequency analysis                          │
│                                                                          │
│  ⚙️ Settings      │  Database management                                │
│                   │  - View database statistics                         │
│                   │  - Upload new JSON files                            │
│                   │  - Automatic duplicate detection                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Features

- **Dark Theme** - Modern dark UI, easy on the eyes
- **RTL Support** - Full Hebrew/Arabic text support
- **Responsive** - Works on mobile and desktop
- **Real-time Charts** - Interactive Chart.js visualizations
- **Export** - Download data as CSV/JSON

---

## AI Search

Ask questions about your chat data in natural language (Hebrew or English).

### Setup AI Provider (Free Options)

#### Option 1: Ollama (Recommended - 100% Local & Free)

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

#### Option 2: Groq (Free API Tier)

```bash
# Get free API key from https://console.groq.com
export GROQ_API_KEY="your_api_key"
```

#### Option 3: Google Gemini (Free API Tier)

```bash
# Get free API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key"
```

### AI Search Examples

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🤖 AI Search - Natural Language Queries                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Query: "מי שלח הכי הרבה הודעות?"                                       │
│  Answer: המשתמש הפעיל ביותר הוא דני עם 5,432 הודעות                     │
│                                                                          │
│  Query: "מתי היו הכי הרבה הודעות?"                                      │
│  Answer: היום הפעיל ביותר היה 15.03.2024 עם 342 הודעות                  │
│                                                                          │
│  Query: "Who mentioned @admin the most?"                                 │
│  Answer: User "Mike" mentioned @admin 47 times                           │
│                                                                          │
│  Query: "הראה הודעות עם קישורים מהשבוע האחרון"                          │
│  Answer: נמצאו 23 הודעות עם קישורים...                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### AI Search API

```python
from ai_search import AISearchEngine

# Initialize with Ollama (local)
ai = AISearchEngine('telegram.db', provider='ollama')

# Or with Groq
ai = AISearchEngine('telegram.db', provider='groq', api_key='your_key')

# Search
result = ai.search("מי הכי פעיל בלילה?")
print(result['answer'])  # Natural language answer
print(result['sql'])     # Generated SQL query
print(result['results']) # Raw data
```

---

## Database Updates

Update your database with new JSON exports without losing existing data.

### Via Web UI

1. Go to **Settings** page in the dashboard
2. Drag & drop your new `result.json` file
3. Wait for processing (duplicate detection automatic)
4. See summary of new messages added

### Via CLI

```bash
# Update existing database with new JSON
python indexer.py new_export.json --db telegram.db --update

# What happens:
# 1. Loads existing message IDs into Bloom filter (O(n))
# 2. For each message in JSON:
#    - Check if exists using Bloom filter (O(1))
#    - Only insert if new
# 3. Re-index FTS if needed
# 4. Report: X new messages, Y duplicates skipped
```

### Incremental Update Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INCREMENTAL UPDATE PROCESS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Existing DB                    New JSON                                 │
│  ┌─────────────┐               ┌─────────────┐                          │
│  │ msg_1 ✓     │               │ msg_1       │ → Skip (duplicate)       │
│  │ msg_2 ✓     │               │ msg_2       │ → Skip (duplicate)       │
│  │ msg_3 ✓     │               │ msg_5  NEW  │ → Insert                 │
│  │ msg_4 ✓     │               │ msg_6  NEW  │ → Insert                 │
│  └─────────────┘               └─────────────┘                          │
│         │                             │                                  │
│         │      Bloom Filter           │                                  │
│         │      ┌───────────┐          │                                  │
│         └─────▶│ O(1) test │◀─────────┘                                  │
│                └───────────┘                                             │
│                                                                          │
│  Result: Only msg_5 and msg_6 added (fast!)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Telegram JSON Export (result.json)                      │    │
│  │  ├── messages[]                                          │    │
│  │  │   ├── id, date, from, text                           │    │
│  │  │   ├── reply_to_message_id                            │    │
│  │  │   └── text_entities[] (links, mentions)              │    │
│  │  └── ...                                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXER (indexer.py)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Batch     │  │   Bloom     │  │   Reply     │              │
│  │  Processing │  │   Filter    │  │   Graph     │              │
│  │  (1000/tx)  │  │ (Dedup O(1))│  │  Builder    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite DATABASE                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  messages          │  FTS5 Index      │  reply_graph    │    │
│  │  ├── id (PK)       │  ├── text_plain  │  ├── parent_id  │    │
│  │  ├── text_plain    │  └── from_name   │  └── child_id   │    │
│  │  ├── from_id       │                  │                 │    │
│  │  ├── date_unixtime │  entities        │  threads        │    │
│  │  └── ...           │  ├── links       │  └── messages   │    │
│  │                    │  └── mentions    │                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   SEARCH    │  │  ANALYZER   │  │   VECTOR    │
│ (search.py) │  │(analyzer.py)│  │  (optional) │
│             │  │             │  │             │
│ • FTS5+BM25 │  │ • Top-K     │  │ • FAISS     │
│ • Fuzzy     │  │ • LCS       │  │ • Semantic  │
│ • Threads   │  │ • Rank Tree │  │ • Clustering│
│ • LRU Cache │  │ • Percentile│  │             │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Data Flow

```
JSON Message                 Database Tables              Search/Analytics
───────────                  ───────────────              ────────────────

{                           ┌─────────────┐
  "id": 548795,       ───▶  │  messages   │  ───▶  Full-text search
  "text": "שלום",           └─────────────┘        User filtering
  "from": "User1",                                 Date range queries
  "from_id": "user123", ─▶  ┌─────────────┐
  "date_unixtime": ...,     │   users     │  ───▶  Top users (Heap)
                            └─────────────┘        User rank (Rank Tree)
  "text_entities": [
    {"type": "link", ────▶  ┌─────────────┐
     "text": "url"}         │  entities   │  ───▶  Link analysis
  ],                        └─────────────┘        Mention network

  "reply_to_message_id" ─▶  ┌─────────────┐
                            │ reply_graph │  ───▶  Thread DFS/BFS
}                           └─────────────┘        Conversation view
```

### File Structure

```
telegram/
│
├── dashboard.py        # 🌐 Web Dashboard (Flask)
│   └── Routes: /, /users, /chat, /search, /moderation, /settings
│   └── API: /api/overview, /api/users, /api/search, /api/update, etc.
│
├── ai_search.py        # 🤖 AI-Powered Search
│   └── AISearchEngine class
│       ├── Natural language to SQL
│       ├── Ollama/Groq/Gemini providers
│       └── Hebrew/English support
│
├── indexer.py          # JSON → SQLite indexer
│   ├── OptimizedIndexer class
│   │   ├── Batch processing (100x faster)
│   │   ├── Bloom filter (duplicate detection)
│   │   └── Graph builder (reply threads)
│   └── IncrementalIndexer class
│       ├── Update existing database
│       ├── Bloom filter duplicate check
│       └── Only insert new messages
│
├── search.py           # Search interface
│   └── TelegramSearch class
│       ├── FTS5 full-text search
│       ├── Fuzzy trigram search
│       ├── LRU query cache
│       └── DFS/BFS thread traversal
│
├── analyzer.py         # Analytics & statistics
│   └── TelegramAnalyzer class
│       ├── LCS similar messages
│       ├── Heap-based Top-K
│       ├── Selection percentiles
│       ├── Rank Tree queries
│       └── Bucket Sort histograms
│
├── data_structures.py  # Core data structures
│   ├── BloomFilter     # O(1) membership test
│   ├── Trie            # O(k) prefix search
│   ├── LRUCache        # O(1) caching
│   ├── ReplyGraph      # DFS/BFS traversal
│   └── TrigramIndex    # Fuzzy matching
│
├── algorithms.py       # Course algorithms
│   ├── LCS             # Similar message detection
│   ├── TopK (Heap)     # Efficient ranking
│   ├── Selection       # O(n) percentiles
│   ├── RankTree        # O(log n) rank queries
│   └── BucketSort      # Time histograms
│
├── templates/          # 🎨 HTML Templates
│   ├── index.html      # Overview dashboard
│   ├── users.html      # User leaderboard
│   ├── chat.html       # Telegram-like chat view
│   ├── search.html     # Search interface
│   ├── moderation.html # Content analytics
│   └── settings.html   # Settings & DB update
│
├── static/             # 📁 Static assets
│   ├── css/style.css   # Dashboard styles
│   └── js/dashboard.js # Dashboard scripts
│
├── vector_search.py    # Optional: Semantic search
│   └── VectorSearch class (requires FAISS)
│
├── schema.sql          # Database schema
└── telegram.db         # SQLite database (created)
```

---

## Usage Guide

### Web Dashboard (Recommended)

```bash
# Start the dashboard
python dashboard.py

# Custom port
python dashboard.py --port 8080

# Custom database
python dashboard.py --db my_chat.db
```

### Indexing

```bash
# Basic indexing
python indexer.py result.json

# Custom database name
python indexer.py result.json --db my_chat.db

# With trigram index (for fuzzy search)
python indexer.py result.json --build-trigrams

# Larger batch size (faster for big files)
python indexer.py result.json --batch-size 5000

# Update existing database with new JSON (incremental)
python indexer.py new_export.json --db telegram.db --update
```

### Searching

```bash
# Basic search (Hebrew supported)
python search.py "שלום"

# Search with filters
python search.py "מילה" --user user123456 --limit 50

# Date range
python search.py "חדשות" --from-date 2024-01-01 --to-date 2024-12-31

# Fuzzy search (finds typos)
python search.py "שלמ" --fuzzy --threshold 0.3

# View conversation thread
python search.py --thread 548795

# List all links
python search.py --list-links

# List all mentions
python search.py --list-mentions
```

### Analytics

```bash
# General statistics
python analyzer.py --stats

# Top users (Heap-based O(n log k))
python analyzer.py --top-users --limit 10

# Hourly activity
python analyzer.py --hourly

# Daily activity
python analyzer.py --daily

# Top words
python analyzer.py --words --limit 30

# Top domains
python analyzer.py --domains

# Find similar messages (LCS algorithm)
python analyzer.py --similar --threshold 0.7

# Find reposts
python analyzer.py --reposts

# Message length percentiles (Selection algorithm)
python analyzer.py --percentiles

# Response time percentiles
python analyzer.py --response-times

# User rank (Rank Tree O(log n))
python analyzer.py --user-rank user123456

# Get user at rank #5
python analyzer.py --rank 5

# Activity histogram (Bucket Sort)
python analyzer.py --histogram --bucket-size 86400

# Export as JSON
python analyzer.py --stats --json > stats.json
```

---

## Algorithms

### Algorithm Complexity Comparison

```
┌────────────────────┬─────────────────┬─────────────────┬─────────────┐
│     Operation      │  Naive Method   │  Our Algorithm  │ Improvement │
├────────────────────┼─────────────────┼─────────────────┼─────────────┤
│ Top-K users        │ O(n log n) sort │ O(n log k) heap │   ~10x      │
│ Find median        │ O(n log n) sort │ O(n) selection  │   ~5x       │
│ User rank query    │ O(n) scan       │ O(log n) tree   │   ~100x     │
│ Duplicate check    │ O(n) lookup     │ O(1) bloom      │   ~1000x    │
│ Similar messages   │ O(n²m²) naive   │ O(n²m) LCS+DP   │   ~10x      │
│ Time histogram     │ O(n log n) sort │ O(n+k) bucket   │   ~5x       │
│ Thread traversal   │ O(n) repeated   │ O(V+E) DFS/BFS  │   ~10x      │
└────────────────────┴─────────────────┴─────────────────┴─────────────┘
```

### 1. LCS (Longest Common Subsequence)

**Purpose:** Find similar/duplicate messages

```
String 1: "שלום לכולם מה קורה"
String 2: "שלום לכולם מה נשמע"
                          ↓
LCS:      "שלום לכולם מה "
Similarity: 77.78%
```

**Algorithm:**
```
┌───┬───┬───┬───┬───┬───┐
│   │ ∅ │ A │ B │ C │ D │   DP Table
├───┼───┼───┼───┼───┼───┤
│ ∅ │ 0 │ 0 │ 0 │ 0 │ 0 │   dp[i][j] = length of LCS
│ A │ 0 │ 1 │ 1 │ 1 │ 1 │   for first i and j chars
│ C │ 0 │ 1 │ 1 │ 2 │ 2 │
│ B │ 0 │ 1 │ 2 │ 2 │ 2 │   Time:  O(m × n)
│ D │ 0 │ 1 │ 2 │ 2 │ 3 │   Space: O(min(m,n))
└───┴───┴───┴───┴───┴───┘
```

### 2. Heap-based Top-K

**Purpose:** Find top K items without sorting everything

```
Finding Top 3 from [5,2,8,1,9,3,7,4,6]

Min-Heap (size K=3):

Step 1: [5]           Add 5
Step 2: [2,5]         Add 2
Step 3: [2,5,8]       Add 8 (heap full)
Step 4: [2,5,8]       Skip 1 (< min)
Step 5: [5,9,8]       Replace 2 with 9
Step 6: [5,9,8]       Skip 3 (< min)
Step 7: [7,9,8]       Replace 5 with 7
...
Result: [7,8,9]       Top 3!

Time: O(n log k) vs O(n log n) for full sort
```

### 3. Selection Algorithm (Median of Medians)

**Purpose:** Find k-th element or percentiles in O(n)

```
Find median of [3,1,4,1,5,9,2,6,5,3,5]

┌─────────────────────────────────────────┐
│  Divide into groups of 5:               │
│  [3,1,4,1,5] [9,2,6,5,3] [5]           │
│       ↓           ↓        ↓            │
│  Medians: 3       5        5            │
│       ↓                                 │
│  Median of medians: 5 (pivot)           │
│       ↓                                 │
│  Partition around 5                     │
│  [3,1,4,1,2,3] [5,5,5] [9,6]           │
│       6 elements  3     2               │
│       ↓                                 │
│  Median is at position 5 → found!       │
└─────────────────────────────────────────┘

Time: O(n) guaranteed (not just average!)
```

### 4. Rank Tree (Order Statistics Tree)

**Purpose:** O(log n) rank queries

```
AVL Tree with size augmentation:

           ┌───────────────┐
           │  150 (size=5) │
           └───────┬───────┘
          ┌────────┴────────┐
    ┌─────┴─────┐     ┌─────┴─────┐
    │ 100 (s=2) │     │ 250 (s=2) │
    └─────┬─────┘     └─────┬─────┘
    ┌─────┴              ┌──┴
┌───┴───┐            ┌───┴───┐
│50 (1) │            │300 (1)│
└───────┘            └───────┘

select(3) → 150  (3rd smallest)
rank(150) → 3    (rank of 150)

Time: O(log n) for both operations
```

### 5. Bucket Sort (Time Histograms)

**Purpose:** O(n+k) time-based grouping

```
Messages with timestamps:
[1000, 1500, 2500, 1200, 3000]

Bucket size: 1000 seconds

┌─────────┬─────────┬─────────┬─────────┐
│ 0-1000  │1000-2000│2000-3000│3000-4000│
├─────────┼─────────┼─────────┼─────────┤
│         │ 1000    │  2500   │  3000   │
│         │ 1500    │         │         │
│         │ 1200    │         │         │
├─────────┼─────────┼─────────┼─────────┤
│ Count:0 │ Count:3 │ Count:1 │ Count:1 │
└─────────┴─────────┴─────────┴─────────┘

Time: O(n + k) where k = number of buckets
```

### 6. DFS/BFS Thread Traversal

**Purpose:** Reconstruct conversation threads

```
Reply Graph:

    [1] Original message
     │
     ├──[2] Reply to 1
     │   │
     │   ├──[4] Reply to 2
     │   │
     │   └──[5] Reply to 2
     │
     └──[3] Reply to 1

DFS order: [1, 2, 4, 5, 3]  (deep first)
BFS order: [1, 2, 3, 4, 5]  (level by level)

With depth info:
  [1] depth=0
    [2] depth=1
      [4] depth=2
      [5] depth=2
    [3] depth=1

Time: O(V + E)
```

---

## API Reference

### Dashboard REST API

The web dashboard exposes a REST API for all operations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REST API ENDPOINTS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GET  /api/overview           Overview statistics                        │
│       ?timeframe=month        (today|yesterday|week|month|year|all)      │
│                                                                          │
│  GET  /api/users              User leaderboard                           │
│       ?timeframe=month        Timeframe filter                           │
│       &limit=100              Max users                                  │
│                                                                          │
│  GET  /api/user/<user_id>     User details                              │
│       ?timeframe=month        Includes hourly activity                   │
│                                                                          │
│  GET  /api/search             Full-text search                           │
│       ?q=search_term          Search query                               │
│       &timeframe=all          Timeframe filter                           │
│       &limit=20&offset=0      Pagination                                 │
│                                                                          │
│  POST /api/ai/search          AI-powered search                          │
│       {"query": "..."}        Natural language query                     │
│                                                                          │
│  GET  /api/chat/messages      Chat messages                              │
│       ?limit=50&offset=0      Pagination                                 │
│       &user_id=...            Filter by user                             │
│       &from_date=...          Date range                                 │
│                                                                          │
│  GET  /api/chat/thread/<id>   Get conversation thread                    │
│                               Returns full thread with DFS               │
│                                                                          │
│  GET  /api/top/domains        Top shared domains                         │
│  GET  /api/top/mentions       Top mentioned users                        │
│  GET  /api/top/words          Most frequent words                        │
│                                                                          │
│  POST /api/update             Update database with JSON                  │
│       (multipart form)        File upload                                │
│                                                                          │
│  GET  /api/db/stats           Database statistics                        │
│                               Size, counts, date range                   │
│                                                                          │
│  GET  /api/export/users       Export users as CSV                        │
│  GET  /api/export/messages    Export messages as CSV                     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                    ALGORITHM-POWERED ENDPOINTS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GET  /api/similar/<id>       Find similar messages (LCS algorithm)      │
│       ?threshold=0.7          Similarity threshold                       │
│       ?limit=10               Max results                                │
│       Complexity: O(n*m)      n=sample, m=avg length                     │
│                                                                          │
│  GET  /api/analytics/similar  Find all similar pairs in DB               │
│       ?threshold=0.8          Similarity threshold                       │
│       Algorithm: LCS          O(n² * m) with early termination           │
│                                                                          │
│  GET  /api/user/rank/<id>     Get user rank (RankTree)                   │
│       Complexity: O(log n)    vs O(n) SQL scan                           │
│                                                                          │
│  GET  /api/user/by-rank/<k>   Get k-th ranked user (RankTree)            │
│       Algorithm: select(k)    O(log n)                                   │
│                                                                          │
│  GET  /api/analytics/histogram Activity histogram (Bucket Sort)          │
│       ?bucket=86400           Bucket size in seconds                     │
│       Complexity: O(n + k)    k=number of buckets                        │
│                                                                          │
│  GET  /api/analytics/percentiles Message length stats (Selection)        │
│       Algorithm: Quickselect  O(n) guaranteed                            │
│       Returns: min,max,median,p25,p75,p90,p95,p99                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### TelegramSearch

```python
from search import TelegramSearch

with TelegramSearch('telegram.db') as search:
    # Full-text search
    results = search.search("שלום", limit=50)

    # With filters
    results = search.search(
        "מילה",
        user_id="user123",
        from_date=1704067200,  # Unix timestamp
        to_date=1735689600,
        has_links=True
    )

    # Fuzzy search
    results = search.fuzzy_search("שלמ", threshold=0.3)

    # Get thread (DFS)
    thread = search.get_thread_dfs(message_id=548795)

    # Get thread with depth
    thread = search.get_thread_with_depth(message_id=548795)
    # Returns: [(message_dict, depth), ...]

    # Autocomplete usernames
    suggestions = search.autocomplete_user("@user")
```

### TelegramAnalyzer

```python
from analyzer import TelegramAnalyzer

with TelegramAnalyzer('telegram.db') as analyzer:
    # Statistics
    stats = analyzer.get_stats()

    # Top users (Heap-based)
    top_users = analyzer.get_top_users(limit=10)

    # Similar messages (LCS)
    similar = analyzer.find_similar_messages(threshold=0.7)

    # Percentiles (Selection algorithm)
    percentiles = analyzer.get_message_length_stats()
    # Returns: {min, max, median, p25, p75, p90, p95, p99}

    # User rank (Rank Tree)
    rank_info = analyzer.get_user_rank("user123")
    # Returns: {rank, total_users, percentile}

    # Get user by rank
    user = analyzer.get_user_by_rank(5)

    # Histogram (Bucket Sort)
    hist = analyzer.get_activity_histogram(bucket_size=86400)
```

---

## Examples

### Example 1: Find Most Active Hours

```python
from analyzer import TelegramAnalyzer

with TelegramAnalyzer('telegram.db') as analyzer:
    hourly = analyzer.get_hourly_activity()

    # Find peak hour
    peak_hour = max(hourly, key=hourly.get)
    print(f"Most active hour: {peak_hour}:00 ({hourly[peak_hour]} messages)")
```

### Example 2: Detect Spam/Reposts

```python
from analyzer import TelegramAnalyzer

with TelegramAnalyzer('telegram.db') as analyzer:
    reposts = analyzer.find_reposts(threshold=0.9)

    for r in reposts[:10]:
        print(f"Similarity: {r['similarity']:.0%}")
        print(f"  User 1: {r['user_1']}")
        print(f"  User 2: {r['user_2']}")
        print(f"  Text: {r['text_preview'][:50]}...")
```

### Example 3: Conversation Thread Analysis

```python
from search import TelegramSearch

with TelegramSearch('telegram.db') as search:
    # Get full thread
    thread = search.get_thread_with_depth(548795)

    print("Conversation thread:")
    for msg, depth in thread:
        indent = "  " * depth
        print(f"{indent}[{msg['from_name']}]: {msg['text_plain'][:50]}")
```

### Example 4: User Ranking

```python
from analyzer import TelegramAnalyzer

with TelegramAnalyzer('telegram.db') as analyzer:
    # Get rank of specific user
    rank = analyzer.get_user_rank("user123456")
    print(f"Rank: #{rank['rank']} of {rank['total_users']}")
    print(f"Top {rank['percentile']:.1f}%")

    # Get top 3 users
    for i in range(1, 4):
        user = analyzer.get_user_by_rank(i)
        print(f"#{i}: {user['name']} ({user['count']} messages)")
```

---

## Performance

Tested on 100,000 messages:

| Operation | Time |
|-----------|------|
| Indexing | ~10 seconds |
| Full-text search | <10ms |
| Fuzzy search | ~100ms |
| Top-K (k=20) | ~50ms |
| User rank query | <1ms |
| Thread traversal | <5ms |
| Similar messages (1000 sample) | ~2 seconds |

---

## License

MIT License - Free for personal and commercial use.

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and create PR

---

## Troubleshooting

### "Module not found" error
```bash
# Make sure you're in the telegram directory
cd /path/to/telegram
python indexer.py result.json
```

### "Database is locked" error
```bash
# Close any other programs using the database
# Or use a different database name
python indexer.py result.json --db telegram2.db
```

### Hebrew text not displaying correctly
```bash
# Ensure your terminal supports UTF-8
export LANG=en_US.UTF-8
```

---

## Credits

Algorithms implemented from "Data Structures and Introduction to Algorithms" course:
- LCS (Longest Common Subsequence)
- Heap-based Top-K
- Selection Algorithm (Median of Medians)
- Rank Tree (Order Statistics Tree)
- Bucket Sort
- DFS/BFS Graph Traversal
- Bloom Filter
- Trie (Prefix Tree)
