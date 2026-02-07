"""
Advanced Stylometry Analysis Module for Hebrew Text
Detects potential duplicate accounts based on writing style patterns.

Uses:
- sentence-transformers for Hebrew embeddings (writing style fingerprint)
- scikit-learn for DBSCAN clustering + TF-IDF on function words
- Hebrew-specific linguistic features (gender, formality, slang)
"""

import re
import sqlite3
import math
import pickle
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# ==========================================
# HEBREW LINGUISTIC PATTERNS
# ==========================================

# Hebrew character ranges
HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

# Hebrew function words (high frequency, style indicators)
HEBREW_FUNCTION_WORDS = [
    'של', 'את', 'על', 'עם', 'אל', 'מן', 'בין', 'לפני', 'אחרי', 'תחת',
    'אני', 'אתה', 'את', 'הוא', 'היא', 'אנחנו', 'אתם', 'אתן', 'הם', 'הן',
    'זה', 'זאת', 'זו', 'אלה', 'אלו',
    'כי', 'אם', 'או', 'גם', 'רק', 'אבל', 'אלא', 'למרות', 'בגלל', 'כדי',
    'מה', 'מי', 'איפה', 'מתי', 'למה', 'איך', 'כמה',
    'כל', 'הרבה', 'קצת', 'מאוד', 'יותר', 'פחות', 'כמו',
    'לא', 'כן', 'אין', 'יש', 'היה', 'להיות', 'עוד', 'כבר',
]

# Formal vs informal markers
FORMAL_MARKERS = ['אנוכי', 'הנני', 'עליכם', 'בבקשה', 'תודה רבה', 'בכבוד רב', 'לכבוד']
INFORMAL_MARKERS = ['אחי', 'גבר', 'אחלה', 'סבבה', 'יאללה', 'וואלה', 'באסה', 'חחחח', 'חחח', 'לול', 'wtf', 'omg']

# Hebrew slang and expressions
HEBREW_SLANG = [
    'אחלה', 'סבבה', 'יאללה', 'וואלה', 'באסה', 'חבל', 'מגניב', 'אשכרה',
    'חחחח', 'חחח', 'הההה', 'ממממ', 'אההה', 'נו', 'טוב', 'בסדר',
    'פיצוץ', 'משהו', 'כאילו', 'סתם', 'ממש', 'פשוט', 'נורא', 'מלא',
]

# Hebrew acronyms
HEBREW_ACRONYMS = ['בעזהש', 'אכא', 'נלענד', 'תנצבה', 'זצל', 'בס"ד', 'בע"ה', 'אי"ה', 'בל"נ']

# Gender markers in verbs (past tense patterns)
MALE_VERB_ENDINGS = ['תי', 'ת', 'נו', 'תם']  # הלכתי, הלכת, הלכנו
FEMALE_VERB_ENDINGS = ['תי', 'ת', 'נו', 'תן']  # הלכתי, הלכת (female), הלכנו

# Repeated character pattern (emotional expression)
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')

# Word with numbers pattern (l33t speak)
LEET_PATTERN = re.compile(r'\b\w*\d+\w*\b')


class AdvancedStyleFeatures:
    """Enhanced features extracted from a user's messages."""

    def __init__(self, user_id: str, user_name: str):
        self.user_id = user_id
        self.user_name = user_name
        self.message_count = 0

        # === Basic Statistics ===
        self.avg_message_length = 0.0
        self.std_message_length = 0.0
        self.avg_word_length = 0.0
        self.avg_words_per_message = 0.0

        # === Character Ratios ===
        self.hebrew_ratio = 0.0
        self.english_ratio = 0.0
        self.digit_ratio = 0.0
        self.emoji_ratio = 0.0
        self.punctuation_ratio = 0.0

        # === Punctuation Patterns ===
        self.comma_rate = 0.0
        self.period_rate = 0.0
        self.question_rate = 0.0
        self.exclamation_rate = 0.0
        self.ellipsis_rate = 0.0
        self.quote_rate = 0.0

        # === Hebrew-Specific Features ===
        self.formality_score = 0.0  # -1 (informal) to +1 (formal)
        self.slang_rate = 0.0
        self.acronym_rate = 0.0
        self.repeated_chars_rate = 0.0
        self.leet_speak_rate = 0.0

        # === Linguistic Patterns ===
        self.function_word_freq: Dict[str, float] = {}
        self.unique_word_ratio = 0.0
        self.hapax_ratio = 0.0  # Words used only once
        self.short_message_ratio = 0.0
        self.long_message_ratio = 0.0

        # === Time Patterns ===
        self.hour_distribution = np.zeros(24)
        self.weekday_distribution = np.zeros(7)
        self.weekend_ratio = 0.0
        self.night_owl_ratio = 0.0  # Messages between 00:00-06:00

        # === Response Patterns ===
        self.reply_rate = 0.0
        self.avg_response_words = 0.0

        # === N-gram Features ===
        self.char_bigrams: Dict[str, float] = {}
        self.char_trigrams: Dict[str, float] = {}
        self.word_bigrams: Dict[str, float] = {}

        # === Embedding (from sentence-transformers) ===
        self.style_embedding: Optional[np.ndarray] = None

        # === TF-IDF Vector ===
        self.tfidf_vector: Optional[np.ndarray] = None

        # === Combined Feature Vector ===
        self.feature_vector: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'message_count': self.message_count,
            'avg_message_length': round(self.avg_message_length, 2),
            'avg_word_length': round(self.avg_word_length, 2),
            'hebrew_ratio': round(self.hebrew_ratio, 3),
            'english_ratio': round(self.english_ratio, 3),
            'emoji_ratio': round(self.emoji_ratio, 4),
            'formality_score': round(self.formality_score, 2),
            'slang_rate': round(self.slang_rate, 3),
            'question_rate': round(self.question_rate, 3),
            'exclamation_rate': round(self.exclamation_rate, 3),
            'repeated_chars_rate': round(self.repeated_chars_rate, 3),
            'weekend_ratio': round(self.weekend_ratio, 3),
            'night_owl_ratio': round(self.night_owl_ratio, 3),
            'unique_word_ratio': round(self.unique_word_ratio, 3),
        }


class AdvancedStylometryAnalyzer:
    """
    ML-powered stylometry analyzer using:
    - sentence-transformers for Hebrew writing style embeddings
    - scikit-learn for TF-IDF and DBSCAN clustering
    - Hebrew linguistic feature extraction
    """

    def __init__(self, db_path: str = 'telegram.db'):
        self.db_path = db_path
        self.user_features: Dict[int, AdvancedStyleFeatures] = {}
        self.similarity_threshold = 0.85

        # ML components (lazy loaded)
        self._embedding_model = None
        self._tfidf_vectorizer = None
        self._scaler = None

        # Cache directory
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))

    @property
    def embedding_model(self):
        """Lazy load sentence-transformers model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use multilingual model that supports Hebrew well
                # Alternative: 'imvladikon/sentence-transformers-alephbert' for pure Hebrew
                print("Loading Hebrew embedding model...")
                self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("Embedding model loaded.")
            except Exception as e:
                print(f"Could not load embedding model: {e}")
                self._embedding_model = False  # Mark as failed
        return self._embedding_model if self._embedding_model else None

    def get_active_users(self, min_messages: int = 300, days: int = 365) -> List[Tuple[str, str, int]]:
        """Get users active in the last N days with at least min_messages."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_timestamp = int(cutoff_date.timestamp())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use from_id and from_name directly from messages table
        query = """
            SELECT from_id, MAX(from_name) as name, COUNT(*) as msg_count
            FROM messages
            WHERE date_unixtime >= ?
              AND from_id IS NOT NULL
              AND text_plain IS NOT NULL
              AND text_plain != ''
            GROUP BY from_id
            HAVING msg_count >= ?
            ORDER BY msg_count DESC
        """

        cursor.execute(query, (cutoff_timestamp, min_messages))
        users = cursor.fetchall()
        conn.close()

        return users

    def get_user_messages(self, user_id: str, days: int = 365) -> List[Tuple[str, str]]:
        """Get messages for a user (text, date)."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_timestamp = int(cutoff_date.timestamp())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT text_plain, date FROM messages
            WHERE from_id = ? AND date_unixtime >= ?
              AND text_plain IS NOT NULL AND text_plain != ''
            ORDER BY date_unixtime
        """

        cursor.execute(query, (user_id, cutoff_timestamp))
        messages = cursor.fetchall()
        conn.close()

        return messages

    def extract_features(self, user_id: str, user_name: str,
                        messages: List[Tuple[str, str]]) -> AdvancedStyleFeatures:
        """Extract comprehensive stylometric features from user messages."""
        features = AdvancedStyleFeatures(user_id, user_name)
        features.message_count = len(messages)

        if not messages:
            return features

        # Collect all text for analysis
        all_texts = [msg[0] for msg in messages if msg[0]]
        all_text_combined = ' '.join(all_texts)

        # === Basic Statistics ===
        message_lengths = [len(text) for text in all_texts]
        features.avg_message_length = np.mean(message_lengths)
        features.std_message_length = np.std(message_lengths)

        all_words = []
        word_counts_per_msg = []
        for text in all_texts:
            words = text.split()
            all_words.extend(words)
            word_counts_per_msg.append(len(words))

        if all_words:
            word_lengths = [len(w) for w in all_words]
            features.avg_word_length = np.mean(word_lengths)
            features.avg_words_per_message = np.mean(word_counts_per_msg)

        # === Character Ratios ===
        total_chars = len(all_text_combined)
        if total_chars > 0:
            hebrew_chars = len(HEBREW_PATTERN.findall(all_text_combined))
            english_chars = len(ENGLISH_PATTERN.findall(all_text_combined))
            digit_chars = sum(1 for c in all_text_combined if c.isdigit())
            punct_chars = sum(1 for c in all_text_combined if c in '.,!?;:()[]{}')
            emoji_count = len(EMOJI_PATTERN.findall(all_text_combined))

            features.hebrew_ratio = hebrew_chars / total_chars
            features.english_ratio = english_chars / total_chars
            features.digit_ratio = digit_chars / total_chars
            features.punctuation_ratio = punct_chars / total_chars
            features.emoji_ratio = emoji_count / total_chars

        # === Punctuation Patterns ===
        n_msgs = len(messages)
        features.comma_rate = all_text_combined.count(',') / n_msgs
        features.period_rate = all_text_combined.count('.') / n_msgs
        features.question_rate = all_text_combined.count('?') / n_msgs
        features.exclamation_rate = all_text_combined.count('!') / n_msgs
        features.ellipsis_rate = all_text_combined.count('...') / n_msgs
        features.quote_rate = (all_text_combined.count('"') + all_text_combined.count("'")) / n_msgs

        # === Hebrew-Specific Features ===
        text_lower = all_text_combined.lower()

        # Formality score
        formal_count = sum(1 for marker in FORMAL_MARKERS if marker in all_text_combined)
        informal_count = sum(1 for marker in INFORMAL_MARKERS if marker in text_lower)
        total_markers = formal_count + informal_count
        if total_markers > 0:
            features.formality_score = (formal_count - informal_count) / total_markers

        # Slang rate
        slang_count = sum(1 for text in all_texts for slang in HEBREW_SLANG if slang in text)
        features.slang_rate = slang_count / n_msgs

        # Acronym rate
        acronym_count = sum(1 for text in all_texts for acr in HEBREW_ACRONYMS if acr in text)
        features.acronym_rate = acronym_count / n_msgs

        # Repeated characters (emotional expression like חחחח)
        repeated_msgs = sum(1 for text in all_texts if REPEATED_CHARS_PATTERN.search(text))
        features.repeated_chars_rate = repeated_msgs / n_msgs

        # Leet speak rate
        leet_count = sum(len(LEET_PATTERN.findall(text)) for text in all_texts)
        features.leet_speak_rate = leet_count / n_msgs

        # === Linguistic Patterns ===
        # Function word frequency
        word_counter = Counter(w.lower() for w in all_words)
        total_words = len(all_words)
        for fw in HEBREW_FUNCTION_WORDS:
            features.function_word_freq[fw] = word_counter.get(fw, 0) / max(1, total_words)

        # Vocabulary richness
        unique_words = set(w.lower() for w in all_words)
        features.unique_word_ratio = len(unique_words) / max(1, total_words)

        # Hapax legomena (words appearing only once)
        hapax_count = sum(1 for w, c in word_counter.items() if c == 1)
        features.hapax_ratio = hapax_count / max(1, len(unique_words))

        # Message length categories
        features.short_message_ratio = sum(1 for wc in word_counts_per_msg if wc < 5) / n_msgs
        features.long_message_ratio = sum(1 for wc in word_counts_per_msg if wc > 30) / n_msgs

        # === Time Patterns ===
        hour_counts = np.zeros(24)
        weekday_counts = np.zeros(7)
        night_msgs = 0
        weekend_msgs = 0

        for text, date_str in messages:
            try:
                if 'T' in date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')

                hour_counts[dt.hour] += 1
                weekday_counts[dt.weekday()] += 1

                if 0 <= dt.hour < 6:
                    night_msgs += 1
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    weekend_msgs += 1
            except:
                pass

        # Normalize
        if hour_counts.sum() > 0:
            features.hour_distribution = hour_counts / hour_counts.sum()
        if weekday_counts.sum() > 0:
            features.weekday_distribution = weekday_counts / weekday_counts.sum()

        features.weekend_ratio = weekend_msgs / n_msgs
        features.night_owl_ratio = night_msgs / n_msgs

        # === N-gram Features ===
        # Character bigrams
        char_bigram_counter = Counter()
        for text in all_texts:
            clean_text = re.sub(r'\s+', ' ', text.lower())
            for i in range(len(clean_text) - 1):
                bg = clean_text[i:i+2]
                if bg.strip():
                    char_bigram_counter[bg] += 1

        total_bigrams = sum(char_bigram_counter.values())
        if total_bigrams > 0:
            for bg, count in char_bigram_counter.most_common(100):
                features.char_bigrams[bg] = count / total_bigrams

        # Character trigrams
        char_trigram_counter = Counter()
        for text in all_texts:
            clean_text = re.sub(r'\s+', ' ', text.lower())
            for i in range(len(clean_text) - 2):
                tg = clean_text[i:i+3]
                if tg.strip():
                    char_trigram_counter[tg] += 1

        total_trigrams = sum(char_trigram_counter.values())
        if total_trigrams > 0:
            for tg, count in char_trigram_counter.most_common(100):
                features.char_trigrams[tg] = count / total_trigrams

        # Word bigrams
        word_bigram_counter = Counter()
        for text in all_texts:
            words = text.lower().split()
            for i in range(len(words) - 1):
                wb = f"{words[i]} {words[i+1]}"
                word_bigram_counter[wb] += 1

        total_word_bigrams = sum(word_bigram_counter.values())
        if total_word_bigrams > 0:
            for wb, count in word_bigram_counter.most_common(50):
                features.word_bigrams[wb] = count / total_word_bigrams

        # === Generate Style Embedding ===
        if self.embedding_model:
            try:
                # Sample messages for embedding (limit for performance)
                sample_texts = all_texts[:100] if len(all_texts) > 100 else all_texts
                # Combine into a style sample
                style_sample = ' '.join(sample_texts)[:5000]  # Limit length
                features.style_embedding = self.embedding_model.encode(style_sample, show_progress_bar=False)
            except Exception as e:
                print(f"Embedding error for user {user_id}: {e}")

        # === Build Numeric Feature Vector ===
        features.feature_vector = self._build_feature_vector(features)

        return features

    def _build_feature_vector(self, f: AdvancedStyleFeatures) -> np.ndarray:
        """Build normalized feature vector for similarity comparison."""
        vector = [
            # Basic stats (normalized)
            f.avg_message_length / 200,
            f.std_message_length / 100,
            f.avg_word_length / 10,
            f.avg_words_per_message / 20,

            # Character ratios
            f.hebrew_ratio,
            f.english_ratio,
            f.digit_ratio * 10,
            f.emoji_ratio * 100,
            f.punctuation_ratio * 10,

            # Punctuation patterns
            f.comma_rate / 2,
            f.period_rate / 2,
            f.question_rate,
            f.exclamation_rate,
            f.ellipsis_rate * 5,
            f.quote_rate,

            # Hebrew-specific
            f.formality_score,
            f.slang_rate * 5,
            f.acronym_rate * 10,
            f.repeated_chars_rate * 5,
            f.leet_speak_rate * 10,

            # Linguistic
            f.unique_word_ratio,
            f.hapax_ratio,
            f.short_message_ratio,
            f.long_message_ratio,

            # Time patterns
            f.weekend_ratio,
            f.night_owl_ratio * 5,
        ]

        # Add hour distribution (24 values)
        vector.extend(f.hour_distribution.tolist())

        # Add weekday distribution (7 values)
        vector.extend(f.weekday_distribution.tolist())

        # Add top function word frequencies (20 values)
        for fw in HEBREW_FUNCTION_WORDS[:20]:
            vector.append(f.function_word_freq.get(fw, 0) * 100)

        return np.array(vector)

    def calculate_similarity(self, f1: AdvancedStyleFeatures, f2: AdvancedStyleFeatures) -> Tuple[float, Dict]:
        """
        Calculate comprehensive similarity between two users.
        Returns overall score and component breakdown.
        """
        scores = {}

        # 1. Feature vector similarity (cosine)
        if f1.feature_vector is not None and f2.feature_vector is not None:
            v1, v2 = f1.feature_vector, f2.feature_vector
            dot_product = np.dot(v1, v2)
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                scores['feature_cosine'] = float(dot_product / (norm1 * norm2))
            else:
                scores['feature_cosine'] = 0.0
        else:
            scores['feature_cosine'] = 0.0

        # 2. Embedding similarity (if available)
        if f1.style_embedding is not None and f2.style_embedding is not None:
            e1, e2 = f1.style_embedding, f2.style_embedding
            dot_product = np.dot(e1, e2)
            norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if norm1 > 0 and norm2 > 0:
                scores['embedding_cosine'] = float(dot_product / (norm1 * norm2))
            else:
                scores['embedding_cosine'] = 0.0
        else:
            scores['embedding_cosine'] = None

        # 3. Character bigram similarity (Jaccard-like)
        scores['bigram_overlap'] = self._ngram_similarity(f1.char_bigrams, f2.char_bigrams)

        # 4. Trigram similarity
        scores['trigram_overlap'] = self._ngram_similarity(f1.char_trigrams, f2.char_trigrams)

        # 5. Word bigram similarity
        scores['word_bigram_overlap'] = self._ngram_similarity(f1.word_bigrams, f2.word_bigrams)

        # 6. Time pattern similarity (hour distribution)
        if f1.hour_distribution.sum() > 0 and f2.hour_distribution.sum() > 0:
            scores['time_pattern'] = float(np.dot(f1.hour_distribution, f2.hour_distribution))
        else:
            scores['time_pattern'] = 0.0

        # === Threshold-based scoring ===
        # Feature Vector is the most reliable discriminator. Use it as a gate:
        # - Below 94%: heavy penalty (likely different people)
        # - 94-96%: moderate score
        # - Above 96%: bonus (likely same person)

        feature_score = scores['feature_cosine']
        bigram_score = scores['bigram_overlap']

        # Base score from key metrics (feature vector is primary)
        base_score = (
            feature_score * 0.50 +
            bigram_score * 0.30 +
            scores['trigram_overlap'] * 0.10 +
            (scores['embedding_cosine'] * 0.10 if scores['embedding_cosine'] is not None else 0)
        )

        # Apply threshold-based multipliers
        if feature_score >= 0.96:
            # Very high feature similarity - likely same person
            multiplier = 1.15
        elif feature_score >= 0.94:
            # High similarity - possible match
            multiplier = 1.0
        elif feature_score >= 0.90:
            # Moderate similarity - penalize
            multiplier = 0.75
        else:
            # Low similarity - heavy penalty
            multiplier = 0.5

        # Additional penalty if bigrams are low
        if bigram_score < 0.80:
            multiplier *= 0.85
        elif bigram_score >= 0.85:
            multiplier *= 1.05

        overall = base_score * multiplier

        # Cap at 100%
        overall = min(overall, 1.0)

        return overall, scores

    def _ngram_similarity(self, ng1: Dict[str, float], ng2: Dict[str, float]) -> float:
        """Calculate similarity between n-gram distributions."""
        if not ng1 or not ng2:
            return 0.0

        all_ngrams = set(ng1.keys()) | set(ng2.keys())
        if not all_ngrams:
            return 0.0

        intersection = 0.0
        union = 0.0

        for ng in all_ngrams:
            v1 = ng1.get(ng, 0)
            v2 = ng2.get(ng, 0)
            intersection += min(v1, v2)
            union += max(v1, v2)

        if union == 0:
            return 0.0

        return intersection / union

    def cluster_users(self, min_cluster_size: int = 2) -> List[List[int]]:
        """
        Use DBSCAN to automatically cluster users with similar writing styles.
        Returns list of clusters (each cluster is a list of user_ids).
        """
        if len(self.user_features) < 2:
            return []

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("scikit-learn not available for clustering")
            return []

        # Build feature matrix
        user_ids = list(self.user_features.keys())
        feature_matrix = []

        for uid in user_ids:
            f = self.user_features[uid]
            if f.feature_vector is not None:
                # Combine feature vector with embedding if available
                if f.style_embedding is not None:
                    combined = np.concatenate([f.feature_vector, f.style_embedding])
                else:
                    combined = f.feature_vector
                feature_matrix.append(combined)
            else:
                feature_matrix.append(np.zeros(50))  # Fallback

        feature_matrix = np.array(feature_matrix)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)

        # DBSCAN clustering
        # eps: maximum distance between samples in a cluster
        # min_samples: minimum samples to form a cluster
        dbscan = DBSCAN(eps=0.5, min_samples=min_cluster_size, metric='cosine')
        labels = dbscan.fit_predict(features_scaled)

        # Group users by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:  # -1 means noise (no cluster)
                clusters[label].append(user_ids[i])

        return [users for users in clusters.values() if len(users) >= min_cluster_size]

    def analyze_all_users(self, min_messages: int = 300, days: int = 365,
                         progress_callback=None) -> Dict:
        """Analyze all active users and find potential duplicates."""

        # Get active users
        users = self.get_active_users(min_messages, days)
        total_users = len(users)

        if progress_callback:
            progress_callback('users_found', total_users)

        # Extract features for each user
        self.user_features = {}
        for idx, (user_id, user_name, msg_count) in enumerate(users):
            messages = self.get_user_messages(user_id, days)
            features = self.extract_features(user_id, user_name or f"User_{user_id}", messages)
            self.user_features[user_id] = features

            if progress_callback:
                progress_callback('user_processed', idx + 1, total_users, user_name or f"User_{user_id}")

        # Find similar pairs
        if progress_callback:
            progress_callback('comparing', 0)

        similar_pairs = []
        user_ids = list(self.user_features.keys())
        total_comparisons = len(user_ids) * (len(user_ids) - 1) // 2
        comparison_count = 0

        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                uid1, uid2 = user_ids[i], user_ids[j]
                f1, f2 = self.user_features[uid1], self.user_features[uid2]

                similarity, score_breakdown = self.calculate_similarity(f1, f2)

                if similarity >= self.similarity_threshold:
                    similar_pairs.append({
                        'user1': f1.to_dict(),
                        'user2': f2.to_dict(),
                        'similarity': round(similarity * 100, 1),
                        'scores': {k: round(v * 100, 1) if v is not None else None
                                  for k, v in score_breakdown.items()},
                        'details': self._get_similarity_details(f1, f2, score_breakdown)
                    })

                comparison_count += 1
                if progress_callback and comparison_count % 100 == 0:
                    progress_callback('comparing', comparison_count, total_comparisons)

        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        # Run clustering
        clusters = self.cluster_users(min_cluster_size=2)
        cluster_info = []
        for cluster in clusters:
            cluster_users = [self.user_features[uid].to_dict() for uid in cluster]
            cluster_info.append({
                'users': cluster_users,
                'size': len(cluster)
            })

        return {
            'total_users_analyzed': total_users,
            'threshold': self.similarity_threshold * 100,
            'potential_duplicates': len(similar_pairs),
            'pairs': similar_pairs,
            'clusters': cluster_info,
            'all_users': [f.to_dict() for f in self.user_features.values()],
            'embedding_model_used': self.embedding_model is not None,
        }

    def _get_similarity_details(self, f1: AdvancedStyleFeatures, f2: AdvancedStyleFeatures,
                                scores: Dict) -> List[str]:
        """Get human-readable similarity details in Hebrew."""
        details = []

        # High embedding similarity
        if scores.get('embedding_cosine') and scores['embedding_cosine'] > 0.85:
            details.append("סגנון כתיבה דומה מאוד (AI embedding)")

        # Message length
        len_diff = abs(f1.avg_message_length - f2.avg_message_length)
        if len_diff < 15:
            details.append(f"אורך הודעה דומה ({f1.avg_message_length:.0f} vs {f2.avg_message_length:.0f})")

        # Hebrew/English ratio
        heb_diff = abs(f1.hebrew_ratio - f2.hebrew_ratio)
        if heb_diff < 0.1:
            details.append(f"יחס עברית דומה ({f1.hebrew_ratio:.0%} vs {f2.hebrew_ratio:.0%})")

        # Emoji usage
        emoji_diff = abs(f1.emoji_ratio - f2.emoji_ratio)
        if emoji_diff < 0.005 and (f1.emoji_ratio > 0.001 or f2.emoji_ratio > 0.001):
            details.append("שימוש דומה באימוג'י")

        # Formality
        form_diff = abs(f1.formality_score - f2.formality_score)
        if form_diff < 0.3:
            if f1.formality_score > 0.3:
                details.append("שניהם כותבים בסגנון פורמלי")
            elif f1.formality_score < -0.3:
                details.append("שניהם כותבים בסגנון לא פורמלי")

        # Slang usage
        if abs(f1.slang_rate - f2.slang_rate) < 0.1:
            if f1.slang_rate > 0.2:
                details.append("שימוש דומה בסלנג")

        # Repeated characters
        if abs(f1.repeated_chars_rate - f2.repeated_chars_rate) < 0.05:
            if f1.repeated_chars_rate > 0.1:
                details.append("שניהם משתמשים בתווים חוזרים (כמו חחחח)")

        # Time patterns
        if scores.get('time_pattern', 0) > 0.8:
            details.append("דפוס שעות פעילות דומה מאוד")

        # Weekend activity
        weekend_diff = abs(f1.weekend_ratio - f2.weekend_ratio)
        if weekend_diff < 0.1:
            details.append("פעילות דומה בסופ\"ש")

        # Night owl
        if abs(f1.night_owl_ratio - f2.night_owl_ratio) < 0.05:
            if f1.night_owl_ratio > 0.1:
                details.append("שניהם פעילים בשעות הלילה")

        # N-gram overlap
        if scores.get('bigram_overlap', 0) > 0.6:
            details.append("דפוסי אותיות דומים מאוד")

        if scores.get('word_bigram_overlap', 0) > 0.4:
            details.append("צירופי מילים דומים")

        return details


# Singleton instance
_analyzer_instance: Optional[AdvancedStylometryAnalyzer] = None

def get_stylometry_analyzer() -> AdvancedStylometryAnalyzer:
    """Get or create the stylometry analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AdvancedStylometryAnalyzer()
    return _analyzer_instance
