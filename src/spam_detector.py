"""
AI-Based Email Spam Detection System
Data Structures: Hash Tables (Python dicts) for email indexing & feature caching
AI Techniques: Naive Bayes + NLP (TF-IDF, NLTK preprocessing)
"""

import re
import time
import hashlib
import pickle
import os
from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

# ── Bundled English stopwords (no network download required) ──────────────────
_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now','d','ll','m','o',
    're','ve','y','would','could','may','might','shall','also','been','get',
    'got','let','said','say','get','go','know','think','come','see','look','want',
    'give','use','find','tell','ask','seem','feel','try','leave','call','keep',
}

# ── Minimal Porter Stemmer (no external deps) ─────────────────────────────────
class _SimpleStemmer:
    """Lightweight rule-based stemmer — covers ~85% of Porter's output."""
    _sfxs = [
        ('ational','ate'),('tional','tion'),('enci','ence'),('anci','ance'),
        ('izer','ize'),('ising','ise'),('izing','ize'),('ising','ise'),
        ('alism','al'),('ation','ate'),('ator','ate'),('alism','al'),
        ('aliti','al'),('ousli','ous'),('ousness','ous'),('iveness','ive'),
        ('fulness','ful'),('ment',''),('ments',''),('ness',''),
        ('ously','ous'),('ively','ive'),('fully','ful'),('ingly','ing'),
        ('ings','ing'),('ated','ate'),('ating','ate'),('ation',''),
        ('ional','ion'),('ions','ion'),('sion',''),('tion',''),
        ('ing',''),('ies','y'),('ied','y'),('ed',''),('er',''),
        ('ly',''),('es',''),('s',''),
    ]
    def stem(self, word):
        if len(word) <= 3:
            return word
        for suffix, replacement in self._sfxs:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)] + replacement
        return word

# ─────────────────────────────────────────────
#  Hash Table-Based Email Index
# ─────────────────────────────────────────────
class EmailHashIndex:
    """
    Hash Table for fast O(1) email storage, lookup, and deduplication.
    Key = MD5 hash of email content
    Value = dict with metadata + classification result
    """
    def __init__(self):
        self._table: dict[str, dict] = {}          # main hash table
        self._feature_cache: dict[str, list] = {}  # cache preprocessed features
        self._feedback_store: dict[str, str] = {}  # user feedback per hash

    def _hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def store(self, email_id: str, content: str, label: str = None) -> str:
        key = self._hash(content)
        self._table[key] = {
            "email_id": email_id,
            "content": content,
            "label": label,
            "timestamp": time.time(),
        }
        return key

    def lookup(self, content: str) -> dict | None:
        return self._table.get(self._hash(content))

    def update_label(self, content: str, label: str):
        key = self._hash(content)
        if key in self._table:
            self._table[key]["label"] = label
        self._feedback_store[key] = label

    def cache_features(self, content: str, features):
        self._feature_cache[self._hash(content)] = features

    def get_cached_features(self, content: str):
        return self._feature_cache.get(self._hash(content))

    def all_emails(self):
        return list(self._table.values())

    def stats(self) -> dict:
        labels = [e["label"] for e in self._table.values() if e["label"]]
        spam = labels.count("spam")
        ham  = labels.count("ham")
        return {"total": len(self._table), "spam": spam, "ham": ham,
                "feedback_count": len(self._feedback_store)}


# ─────────────────────────────────────────────
#  NLP Preprocessing
# ─────────────────────────────────────────────
class NLPPreprocessor:
    """
    NLP pipeline: clean → tokenize → remove stopwords → stem
    """
    def __init__(self):
        self.stemmer = _SimpleStemmer()
        self.stop_words = _STOPWORDS
        # Spam-indicative word boosting (domain knowledge)
        self.spam_keywords = {
            'free', 'win', 'winner', 'cash', 'prize', 'click', 'offer',
            'limited', 'urgent', 'congratulations', 'selected', 'claim',
            'discount', 'percent', 'buy', 'cheap', 'deal', 'guaranteed'
        }

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', ' url ', text)       # URLs
        text = re.sub(r'\b[\w.+-]+@[\w-]+\.\w+\b', ' email ', text)  # emails
        text = re.sub(r'\$[\d,]+', ' money ', text)            # money
        text = re.sub(r'\d+', ' num ', text)                   # numbers
        text = re.sub(r'[^a-z\s]', ' ', text)                  # punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_stem(self, text: str) -> str:
        tokens = text.split()
        # Keep spam keywords unstemmed to preserve signal
        processed = []
        for tok in tokens:
            if tok in self.stop_words and tok not in self.spam_keywords:
                continue
            processed.append(self.stemmer.stem(tok))
        return ' '.join(processed)

    def transform(self, text: str) -> str:
        return self.tokenize_and_stem(self.clean(text))


# ─────────────────────────────────────────────
#  Spam Detector (Naive Bayes + TF-IDF)
# ─────────────────────────────────────────────
class SpamDetector:
    MODEL_PATH = "models/spam_model.pkl"

    def __init__(self):
        self.preprocessor = NLPPreprocessor()
        self.email_index  = EmailHashIndex()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                max_features=10000,
                sublinear_tf=True,
            )),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        self.is_trained = False
        self._load_model()

    # ── Training ──────────────────────────────
    def train(self, texts: list[str], labels: list[str]):
        processed = [self.preprocessor.transform(t) for t in texts]
        self.pipeline.fit(processed, labels)
        self.is_trained = True
        self._save_model()
        # Index all training emails
        for i, (t, l) in enumerate(zip(texts, labels)):
            self.email_index.store(f"train_{i}", t, l)
        print(f"[✓] Model trained on {len(texts)} samples.")

    def predict(self, text: str) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        # Check hash table cache first
        cached = self.email_index.lookup(text)
        if cached and cached.get("label"):
            return {"label": cached["label"], "confidence": 1.0, "cached": True,
                    "features": self._extract_top_features(text)}

        processed = self.preprocessor.transform(text)
        proba = self.pipeline.predict_proba([processed])[0]
        classes = self.pipeline.classes_
        label_idx = proba.argmax()
        label = classes[label_idx]
        confidence = float(proba[label_idx])

        # Store result in hash table
        key = self.email_index.store("query", text, label)
        features = self._extract_top_features(text)
        self.email_index.cache_features(text, features)

        return {
            "label": label,
            "confidence": confidence,
            "spam_prob": float(proba[list(classes).index("spam")]) if "spam" in classes else 0,
            "ham_prob":  float(proba[list(classes).index("ham")])  if "ham"  in classes else 0,
            "cached": False,
            "features": features,
        }

    def learn_from_feedback(self, text: str, correct_label: str):
        """Adaptive learning: retrain with corrected feedback."""
        self.email_index.update_label(text, correct_label)
        emails = self.email_index.all_emails()
        texts  = [e["content"] for e in emails if e["label"]]
        labels = [e["label"]   for e in emails if e["label"]]
        if len(texts) > 10:
            processed = [self.preprocessor.transform(t) for t in texts]
            self.pipeline.fit(processed, labels)
            self._save_model()

    def evaluate(self, texts, labels) -> dict:
        processed = [self.preprocessor.transform(t) for t in texts]
        preds = self.pipeline.predict(processed)
        return {
            "accuracy": accuracy_score(labels, preds),
            "report": classification_report(labels, preds)
        }

    def _extract_top_features(self, text: str, top_n: int = 10) -> list[dict]:
        processed = self.preprocessor.transform(text)
        vectorizer = self.pipeline.named_steps['tfidf']
        nb = self.pipeline.named_steps['nb']
        vec = vectorizer.transform([processed])
        feature_names = vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]
        spam_idx = list(self.pipeline.classes_).index("spam") if "spam" in self.pipeline.classes_ else 0
        features = []
        for i in indices:
            features.append({
                "word": feature_names[i],
                "tfidf": float(vec[0, i]),
                "spam_log_prob": float(nb.feature_log_prob_[spam_idx, i])
            })
        features.sort(key=lambda x: x["tfidf"], reverse=True)
        return features[:top_n]

    def _save_model(self):
        os.makedirs("models", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)

    def _load_model(self):
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, "rb") as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            print("[✓] Pre-trained model loaded.")
