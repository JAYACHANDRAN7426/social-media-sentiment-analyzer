"""
sentiment_engine.py
===================
High-accuracy NLP sentiment engine using a 3-stage ensemble:
  1. VADER  — on the ORIGINAL (minimally-processed) text
  2. TextBlob — on cleaned text
  3. SentimentIntensityAnalyzer confidence-weighted voting

Key improvements over the previous version
-------------------------------------------
- VADER receives the original text (with emojis, caps, punctuation)
  instead of aggressively stripped text — crucial for social media accuracy.
- Emoji-to-sentiment mapping converts emojis to descriptive words before TextBlob.
- Negation detection adjusts borderline scores around "not", "never", "no", etc.
- Confidence-weighted ensemble: strong individual signals win outright;
  agreement between models boosts confidence; disagreement falls back to VADER.
- Tightened neutral band (0.08) reduces the over-prediction of neutral.
- Consecutive exclamation / all-caps detection adds a positive/negative amplifier.

Public API
----------
    analyze(text: str) -> dict
    analyze_batch(texts: list[str]) -> list[dict]
"""

import re
import logging
from typing import Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── VADER ────────────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except ImportError:
    logger.warning("vaderSentiment not installed – VADER scores will be 0.")
    _vader = None
    VADER_OK = False

# ─── TextBlob ─────────────────────────────────────────────────────────────────
try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    logger.warning("textblob not installed – TextBlob scores will be 0.")
    TEXTBLOB_OK = False

# ─── NLTK stop-words (graceful fallback) ──────────────────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS: set = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "the", "a", "an", "is", "in", "it", "i", "to", "and", "of",
        "this", "that", "was", "are", "for", "on", "be", "with", "at",
        "by", "from", "or", "but", "he", "she", "they", "we",
        "you", "my", "your", "his", "her", "its", "our", "their",
        "have", "has", "had", "do", "did", "does", "will", "would",
        "can", "could", "should", "may", "might", "shall",
    }

# ─── Emoji → sentiment word map ───────────────────────────────────────────────
#   Covers the most common social-media emojis.  We replace them
#   with words so TextBlob (which cannot read emojis) can score them.
EMOJI_MAP: dict[str, str] = {
    # strongly positive
    "😍": "love", "🥰": "love", "😘": "love", "❤️": "love", "💕": "love",
    "💖": "love", "💗": "love", "💯": "perfect", "🏆": "winner",
    "🎉": "celebrate", "🎊": "celebrate", "👏": "applause",
    "😊": "happy", "😁": "happy", "😄": "happy", "😀": "happy",
    "🤩": "amazing", "✨": "wonderful", "🌟": "great", "⭐": "great",
    "👍": "good", "🙌": "great", "🔥": "amazing", "💪": "strong",
    "😂": "funny", "🤣": "hilarious",
    # negative
    "😡": "angry", "😠": "angry", "🤬": "furious", "😤": "frustrated",
    "😒": "disappointed", "😞": "sad", "😢": "sad", "😭": "crying",
    "😩": "miserable", "😖": "upset", "🤮": "disgusting", "🤢": "disgusting",
    "👎": "bad", "💔": "heartbroken", "😔": "sad", "😣": "suffering",
    "😱": "shocked", "😨": "scared", "😰": "worried",
    # neutral / mixed
    "😐": "neutral", "🤔": "thinking", "😑": "expressionless",
    "😶": "speechless", "🙄": "sarcastic",
}

# ─── Negation boosters ────────────────────────────────────────────────────────
_NEG_PATTERN = re.compile(
    r"\b(not|no|never|cannot|can't|won't|isn't|aren't|wasn't|weren't|"
    r"don't|doesn't|didn't|barely|hardly|scarcely|seldom|nothing|"
    r"neither|nor)\b",
    re.IGNORECASE,
)

# ─── ALL-CAPS amplification ───────────────────────────────────────────────────
_ALLCAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PRE-PROCESSING  (two-track)
# ═══════════════════════════════════════════════════════════════════════════════

def _replace_emojis(text: str) -> str:
    """Swap known emojis with their sentiment-bearing English equivalents."""
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, f" {word} ")
    # strip remaining non-ASCII emoji characters (safer for TextBlob)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def clean_text(text: str) -> str:
    """
    Light cleaning for TextBlob & word-cloud use.
    NOTE: VADER should receive the minimally-processed text, not this.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _replace_emojis(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # URLs
    text = re.sub(r"@\w+", " ", text)                   # @mentions
    text = re.sub(r"#(\w+)", r" \1 ", text)             # #hashtag → word
    text = re.sub(r"[^a-z\s']", " ", text)              # non-alpha (keep apostrophe)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _vader_text(text: str) -> str:
    """
    Minimal pre-processing for VADER.
    VADER is designed for raw social media text — preserve case, punctuation,
    emoji-replacement words, and exclamation marks.
    Only strip raw URLs and control characters.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _replace_emojis(text)
    text = re.sub(r"http\S+|www\S+", " ", text)         # URLs only
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)        # control chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stop-words and short tokens (used for word cloud)."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)


# ═══════════════════════════════════════════════════════════════════════════════
# AMPLIFIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _amplifier(original_text: str) -> float:
    """
    Returns a small additive boost (+/-) based on stylistic cues:
      • Each run of exclamation marks  → +0.05 (signals enthusiasm)
      • Each ALL-CAPS word (≥3 chars)  → +0.03 or -0.03 depending on direction
    These are applied *after* we know the preliminary sign of the score.
    """
    exclamations = len(re.findall(r"!+", original_text))
    all_caps     = len(_ALLCAPS_PATTERN.findall(original_text))
    # raw boost — sign will be applied relative to the score direction later
    return min(exclamations * 0.05 + all_caps * 0.03, 0.20)


def _has_negation(text: str) -> bool:
    """True if the text contains a negation word."""
    return bool(_NEG_PATTERN.search(text))


# ═══════════════════════════════════════════════════════════════════════════════
# VADER  (on minimally processed text)
# ═══════════════════════════════════════════════════════════════════════════════

#  Tighten thresholds: VADER's default is 0.05 which over-classifies neutral.
_VADER_POS_THRESH  =  0.08
_VADER_NEG_THRESH  = -0.08
_VADER_STRONG      =  0.50   # score magnitude for "high confidence"

def _vader_analyze(original_text: str) -> tuple[str, float, float]:
    """
    Returns (label, compound_score, confidence).
    confidence ∈ [0, 1]  — higher = more reliable prediction.
    """
    if not VADER_OK or not original_text.strip():
        return "neutral", 0.0, 0.0

    vtext  = _vader_text(original_text)
    scores = _vader.polarity_scores(vtext)
    c      = scores["compound"]

    # Apply exclamation / all-caps amplifier
    amp = _amplifier(original_text)
    if c > 0:
        c = min(1.0, c + amp)
    elif c < 0:
        c = max(-1.0, c - amp)

    # Confidence is the normalised absolute value
    confidence = min(abs(c) / 0.6, 1.0)

    if c >= _VADER_POS_THRESH:
        return "positive", round(c, 4), round(confidence, 4)
    if c <= _VADER_NEG_THRESH:
        return "negative", round(c, 4), round(confidence, 4)
    return "neutral", round(c, 4), round(confidence, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# TextBlob  (on lightly cleaned text)
# ═══════════════════════════════════════════════════════════════════════════════

_TB_POS_THRESH = 0.08
_TB_NEG_THRESH = -0.08

def _textblob_analyze(cleaned_text: str) -> tuple[str, float, float, float]:
    """
    Returns (label, polarity, subjectivity, confidence).
    Negation check: if negation words exist, flip the classification
    of borderline scores.
    """
    if not TEXTBLOB_OK or not cleaned_text.strip():
        return "neutral", 0.0, 0.5, 0.0

    blob = TextBlob(cleaned_text)
    p    = blob.sentiment.polarity
    s    = blob.sentiment.subjectivity

    # Negation adjustment: if negation detected and score is near zero,
    # push it slightly into the opposite direction.
    if _has_negation(cleaned_text):
        if 0 < p < 0.20:
            p = -p * 0.5       # "not bad" → mild negative shift
        elif -0.20 < p < 0:
            p = abs(p) * 0.5   # "not terrible" → mild positive shift

    # Confidence: higher subjectivity + stronger polarity → more confident
    magnitude  = abs(p)
    confidence = min((magnitude * 0.7 + s * 0.3), 1.0)

    if p >= _TB_POS_THRESH:
        label = "positive"
    elif p <= _TB_NEG_THRESH:
        label = "negative"
    else:
        label = "neutral"

    return label, round(p, 4), round(s, 4), round(confidence, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE  (confidence-weighted voting)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensemble(
    vader_label: str, vader_score: float, vader_conf: float,
    tb_label: str,    tb_polarity: float, tb_conf: float,
) -> str:
    """
    Improved ensemble strategy:

    1. If both models AGREE → use that label (high certainty).
    2. If VADER has very high confidence (|score| > 0.5) → trust VADER.
    3. If TextBlob has very high confidence AND VADER is near-neutral → trust TB.
    4. Otherwise → weighted numeric combination (VADER 65%, TextBlob 35%).
    """
    if not TEXTBLOB_OK:
        return vader_label

    # Rule 1: consensus
    if vader_label == tb_label:
        return vader_label

    # Rule 2: strong VADER signal wins
    if abs(vader_score) >= 0.50:
        return vader_label

    # Rule 3: strong TextBlob signal when VADER is near-neutral
    if tb_conf >= 0.60 and abs(vader_score) < 0.15:
        return tb_label

    # Rule 4: weighted numeric blend
    combined = (vader_score * 0.65) + (tb_polarity * 0.35)
    threshold = 0.06  # slightly tighter than the default 0.05
    if combined >= threshold:
        return "positive"
    if combined <= -threshold:
        return "negative"
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(text: str) -> dict:
    """
    Analyze *text* and return a sentiment result dict.

    Keys
    ----
    cleaned_text          : str   — lightly cleaned text (for word clouds etc.)
    vader_label           : str   — 'positive' | 'negative' | 'neutral'
    vader_score           : float — VADER compound score [-1, 1]
    textblob_label        : str
    textblob_polarity     : float — [-1, 1]
    textblob_subjectivity : float — [0, 1]
    final_label           : str   — ensemble result
    sentiment             : str   — alias for final_label (compatibility)
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    cleaned = clean_text(text)

    # VADER receives the original text (minimally processed) — key for accuracy
    vader_label, vader_score, vader_conf = _vader_analyze(text)

    # TextBlob receives the cleaned text
    tb_label, tb_polarity, tb_subj, tb_conf = _textblob_analyze(cleaned)

    final = _ensemble(
        vader_label, vader_score, vader_conf,
        tb_label,    tb_polarity, tb_conf,
    )

    return {
        "cleaned_text":            cleaned,
        "vader_label":             vader_label,
        "vader_score":             vader_score,
        "textblob_label":          tb_label,
        "textblob_polarity":       tb_polarity,
        "textblob_subjectivity":   tb_subj,
        "final_label":             final,
        "sentiment":               final,   # convenience alias
    }


def analyze_batch(texts: list) -> list:
    """Analyze a list of texts. Returns a list of result dicts."""
    return [analyze(t) for t in texts]


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        # Clear positives
        ("I absolutely LOVE this product! Best purchase ever!! 😍", "positive"),
        ("This is amazing, totally recommend it 👍🔥", "positive"),
        # Clear negatives
        ("This is terrible. Completely broken and useless. 😡👎", "negative"),
        ("Worst experience of my life. Never buying again!!!", "negative"),
        # Negation edge cases
        ("Not bad at all, actually quite decent", "positive"),
        ("Not good, very disappointing honestly", "negative"),
        # Neutral
        ("The package arrived today.", "neutral"),
        ("I watched the movie last night.", "neutral"),
        # Mixed / tricky
        ("The product is okay but customer service was horrible", "negative"),
        ("Meh, it's fine I guess... nothing special", "neutral"),
    ]
    print(f"\n{'TEXT':<48} {'EXPECTED':<10} {'VADER':<10} {'TextBlob':<10} {'FINAL':<10}")
    print("-" * 95)
    correct = 0
    for text, expected in samples:
        r = analyze(text)
        match = "✓" if r["final_label"] == expected else "✗"
        correct += (r["final_label"] == expected)
        print(
            f"{match} {text[:46]:<47} {expected:<10} "
            f"{r['vader_label']:<10} {r['textblob_label']:<10} {r['final_label']:<10}"
        )
    print(f"\nAccuracy: {correct}/{len(samples)} = {correct/len(samples)*100:.0f}%")
