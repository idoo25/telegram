"""
Stylometry Analysis Module for Hebrew Text
Detects potential duplicate accounts based on writing style patterns.
"""

import re
import sqlite3
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Hebrew character range
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

# Common Hebrew slang and expressions
HEBREW_SLANG = ['אחלה', 'סבבה', 'יאללה', 'וואלה', 'באסה', 'חבל', 'מגניב', 'אשכרה', 'חחחח', 'חחח', 'הההה', 'ממממ']
HEBREW_ACRONYMS = ['בעזהש', 'אכא', 'לול', 'בטח', 'נלענד', 'תנצבה', 'זאת']


class StyleFeatures:
    """Features extracted from a user's messages."""

    def __init__(self, user_id: int, user_name: str):
        self.user_id = user_id
        self.user_name = user_name
        self.message_count = 0

        # Length features
        self.avg_message_length = 0.0
        self.avg_word_length = 0.0
        self.std_message_length = 0.0

        # Character ratios
        self.hebrew_ratio = 0.0
        self.english_ratio = 0.0
        self.digit_ratio = 0.0
        self.emoji_ratio = 0.0

        # Punctuation patterns
        self.comma_rate = 0.0
        self.period_rate = 0.0
        self.question_rate = 0.0
        self.exclamation_rate = 0.0
        self.ellipsis_rate = 0.0  # ...

        # Special patterns
        self.caps_ratio = 0.0
        self.repeated_chars_rate = 0.0  # כןןןןן
        self.slang_rate = 0.0

        # Time patterns (24 hours distribution)
        self.hour_distribution = [0.0] * 24
        self.weekend_ratio = 0.0

        # Word patterns
        self.unique_word_ratio = 0.0
        self.short_message_ratio = 0.0  # < 5 words

        # Top character bigrams (normalized)
        self.char_bigrams: Dict[str, float] = {}

        # Feature vector for similarity calculation
        self.feature_vector: List[float] = []

    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'message_count': self.message_count,
            'avg_message_length': round(self.avg_message_length, 2),
            'avg_word_length': round(self.avg_word_length, 2),
            'hebrew_ratio': round(self.hebrew_ratio, 3),
            'english_ratio': round(self.english_ratio, 3),
            'emoji_ratio': round(self.emoji_ratio, 3),
            'question_rate': round(self.question_rate, 3),
            'exclamation_rate': round(self.exclamation_rate, 3),
            'ellipsis_rate': round(self.ellipsis_rate, 3),
            'repeated_chars_rate': round(self.repeated_chars_rate, 3),
            'weekend_ratio': round(self.weekend_ratio, 3),
            'unique_word_ratio': round(self.unique_word_ratio, 3),
        }


class StylometryAnalyzer:
    """Analyzes writing styles to detect potential duplicate accounts."""

    def __init__(self, db_path: str = 'telegram_data.db'):
        self.db_path = db_path
        self.user_features: Dict[int, StyleFeatures] = {}
        self.similarity_threshold = 0.85  # Adjustable threshold

    def get_active_users(self, min_messages: int = 300, days: int = 365) -> List[Tuple[int, str, int]]:
        """Get users active in the last N days with at least min_messages."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT u.id, u.name, COUNT(m.id) as msg_count
            FROM users u
            JOIN messages m ON u.id = m.sender_id
            WHERE m.date >= ?
            GROUP BY u.id
            HAVING msg_count >= ?
            ORDER BY msg_count DESC
        """

        cursor.execute(query, (cutoff_str, min_messages))
        users = cursor.fetchall()
        conn.close()

        return users

    def get_user_messages(self, user_id: int, days: int = 365) -> List[Tuple[str, str]]:
        """Get messages for a user (text, date)."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT text, date FROM messages
            WHERE sender_id = ? AND date >= ? AND text IS NOT NULL AND text != ''
            ORDER BY date
        """

        cursor.execute(query, (user_id, cutoff_str))
        messages = cursor.fetchall()
        conn.close()

        return messages

    def extract_features(self, user_id: int, user_name: str, messages: List[Tuple[str, str]]) -> StyleFeatures:
        """Extract stylometric features from user messages."""
        features = StyleFeatures(user_id, user_name)
        features.message_count = len(messages)

        if not messages:
            return features

        # Collect statistics
        message_lengths = []
        word_lengths = []
        all_words = []
        unique_words = set()
        short_messages = 0

        hebrew_chars = 0
        english_chars = 0
        digit_chars = 0
        total_chars = 0
        caps_chars = 0

        commas = 0
        periods = 0
        questions = 0
        exclamations = 0
        ellipsis = 0

        repeated_char_msgs = 0
        slang_count = 0
        emoji_count = 0

        hour_counts = [0] * 24
        weekend_msgs = 0

        char_bigram_counter = Counter()

        for text, date_str in messages:
            if not text:
                continue

            # Message length
            msg_len = len(text)
            message_lengths.append(msg_len)
            total_chars += msg_len

            # Word analysis
            words = text.split()
            if len(words) < 5:
                short_messages += 1
            for word in words:
                word_lengths.append(len(word))
                all_words.append(word.lower())
                unique_words.add(word.lower())

            # Character analysis
            hebrew_chars += len(HEBREW_PATTERN.findall(text))
            english_chars += len(ENGLISH_PATTERN.findall(text))
            digit_chars += sum(1 for c in text if c.isdigit())
            caps_chars += sum(1 for c in text if c.isupper())

            # Emoji analysis
            emojis = EMOJI_PATTERN.findall(text)
            emoji_count += len(emojis)

            # Punctuation
            commas += text.count(',')
            periods += text.count('.')
            questions += text.count('?')
            exclamations += text.count('!')
            ellipsis += text.count('...')

            # Repeated characters pattern (like כןןןןן or אההההה)
            if re.search(r'(.)\1{3,}', text):
                repeated_char_msgs += 1

            # Slang detection
            text_lower = text.lower()
            for slang in HEBREW_SLANG:
                if slang in text:
                    slang_count += 1
                    break

            # Time analysis
            try:
                if 'T' in date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
                hour_counts[dt.hour] += 1
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    weekend_msgs += 1
            except:
                pass

            # Character bigrams
            clean_text = re.sub(r'\s+', ' ', text.lower())
            for i in range(len(clean_text) - 1):
                bigram = clean_text[i:i+2]
                if bigram.strip():
                    char_bigram_counter[bigram] += 1

        n_msgs = len(messages)

        # Calculate averages
        if message_lengths:
            features.avg_message_length = sum(message_lengths) / len(message_lengths)
            variance = sum((x - features.avg_message_length) ** 2 for x in message_lengths) / len(message_lengths)
            features.std_message_length = math.sqrt(variance)

        if word_lengths:
            features.avg_word_length = sum(word_lengths) / len(word_lengths)

        # Character ratios
        if total_chars > 0:
            features.hebrew_ratio = hebrew_chars / total_chars
            features.english_ratio = english_chars / total_chars
            features.digit_ratio = digit_chars / total_chars
            features.emoji_ratio = emoji_count / total_chars
            features.caps_ratio = caps_chars / max(1, english_chars)

        # Punctuation rates (per message)
        features.comma_rate = commas / n_msgs
        features.period_rate = periods / n_msgs
        features.question_rate = questions / n_msgs
        features.exclamation_rate = exclamations / n_msgs
        features.ellipsis_rate = ellipsis / n_msgs

        # Special patterns
        features.repeated_chars_rate = repeated_char_msgs / n_msgs
        features.slang_rate = slang_count / n_msgs

        # Time patterns
        total_hour_msgs = sum(hour_counts)
        if total_hour_msgs > 0:
            features.hour_distribution = [h / total_hour_msgs for h in hour_counts]
            features.weekend_ratio = weekend_msgs / n_msgs

        # Word patterns
        if all_words:
            features.unique_word_ratio = len(unique_words) / len(all_words)
        features.short_message_ratio = short_messages / n_msgs

        # Top character bigrams (normalized)
        total_bigrams = sum(char_bigram_counter.values())
        if total_bigrams > 0:
            top_bigrams = char_bigram_counter.most_common(50)
            features.char_bigrams = {bg: count / total_bigrams for bg, count in top_bigrams}

        # Build feature vector for similarity calculation
        features.feature_vector = self._build_feature_vector(features)

        return features

    def _build_feature_vector(self, f: StyleFeatures) -> List[float]:
        """Build normalized feature vector for similarity comparison."""
        vector = [
            f.avg_message_length / 100,  # Normalize to ~1
            f.avg_word_length / 10,
            f.hebrew_ratio,
            f.english_ratio,
            f.emoji_ratio * 10,  # Scale up small values
            f.question_rate,
            f.exclamation_rate,
            f.ellipsis_rate * 5,
            f.repeated_chars_rate * 10,
            f.weekend_ratio,
            f.unique_word_ratio,
            f.short_message_ratio,
            f.caps_ratio,
            f.slang_rate,
            f.comma_rate,
            f.period_rate,
        ]

        # Add hour distribution (24 values)
        vector.extend(f.hour_distribution)

        return vector

    def calculate_similarity(self, f1: StyleFeatures, f2: StyleFeatures) -> float:
        """Calculate cosine similarity between two feature vectors."""
        v1 = f1.feature_vector
        v2 = f2.feature_vector

        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        # Also compare character bigrams (Jaccard-like)
        bigram_sim = self._compare_bigrams(f1.char_bigrams, f2.char_bigrams)

        # Weighted combination
        return 0.7 * cosine_sim + 0.3 * bigram_sim

    def _compare_bigrams(self, bg1: Dict[str, float], bg2: Dict[str, float]) -> float:
        """Compare character bigram distributions."""
        if not bg1 or not bg2:
            return 0.0

        all_bigrams = set(bg1.keys()) | set(bg2.keys())
        if not all_bigrams:
            return 0.0

        # Calculate similarity based on shared bigrams
        intersection = 0.0
        union = 0.0

        for bg in all_bigrams:
            v1 = bg1.get(bg, 0)
            v2 = bg2.get(bg, 0)
            intersection += min(v1, v2)
            union += max(v1, v2)

        if union == 0:
            return 0.0

        return intersection / union

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
                progress_callback('user_processed', idx + 1, total_users, user_name)

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

                similarity = self.calculate_similarity(f1, f2)

                if similarity >= self.similarity_threshold:
                    similar_pairs.append({
                        'user1': f1.to_dict(),
                        'user2': f2.to_dict(),
                        'similarity': round(similarity * 100, 1),
                        'details': self._get_similarity_details(f1, f2)
                    })

                comparison_count += 1
                if progress_callback and comparison_count % 100 == 0:
                    progress_callback('comparing', comparison_count, total_comparisons)

        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'total_users_analyzed': total_users,
            'threshold': self.similarity_threshold * 100,
            'potential_duplicates': len(similar_pairs),
            'pairs': similar_pairs,
            'all_users': [f.to_dict() for f in self.user_features.values()]
        }

    def _get_similarity_details(self, f1: StyleFeatures, f2: StyleFeatures) -> List[str]:
        """Get human-readable similarity details."""
        details = []

        # Message length similarity
        len_diff = abs(f1.avg_message_length - f2.avg_message_length)
        if len_diff < 10:
            details.append(f"אורך הודעה דומה ({f1.avg_message_length:.0f} vs {f2.avg_message_length:.0f})")

        # Hebrew/English ratio
        heb_diff = abs(f1.hebrew_ratio - f2.hebrew_ratio)
        if heb_diff < 0.1:
            details.append(f"יחס עברית/אנגלית דומה ({f1.hebrew_ratio:.0%} vs {f2.hebrew_ratio:.0%})")

        # Emoji usage
        emoji_diff = abs(f1.emoji_ratio - f2.emoji_ratio)
        if emoji_diff < 0.01:
            details.append("שימוש דומה באימוג'י")

        # Question marks
        q_diff = abs(f1.question_rate - f2.question_rate)
        if q_diff < 0.1:
            details.append("שימוש דומה בסימני שאלה")

        # Weekend activity
        weekend_diff = abs(f1.weekend_ratio - f2.weekend_ratio)
        if weekend_diff < 0.1:
            details.append("פעילות דומה בסופ\"ש")

        # Repeated characters
        if abs(f1.repeated_chars_rate - f2.repeated_chars_rate) < 0.05:
            if f1.repeated_chars_rate > 0.1:
                details.append("שניהם משתמשים בתווים חוזרים (כמו כןןןןן)")

        # Time patterns
        hour_sim = sum(min(h1, h2) for h1, h2 in zip(f1.hour_distribution, f2.hour_distribution))
        if hour_sim > 0.7:
            details.append("דפוס שעות פעילות דומה")

        return details


# Singleton instance
_analyzer_instance: Optional[StylometryAnalyzer] = None

def get_stylometry_analyzer() -> StylometryAnalyzer:
    """Get or create the stylometry analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = StylometryAnalyzer()
    return _analyzer_instance
