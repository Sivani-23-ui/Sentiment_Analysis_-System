"""NLP preprocessing: lowercase -> remove URLs/mentions -> clean -> tokenize -> stopwords -> lemmatize.
Works with NLTK if its data is available, otherwise falls back to built-in rules (so it never crashes offline)."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")

# Emojis carry strong sentiment, so we turn a few into words instead of just deleting them.
_EMOJI = {
    "😀": " good ", "😊": " good ", "😍": " love ", "👍": " good ", "❤": " love ", "🔥": " great ",
    "😡": " bad ", "😠": " bad ", "👎": " bad ", "😢": " sad ", "😭": " sad ", "🤮": " awful ",
    ":)": " good ", ":-)": " good ", ":(": " sad ", ":-(": " sad ",
}

NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "neither", "cannot"}

_FALLBACK_STOP = set("""a about above after again all am an and any are as at be because been before being below between both
but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just me more most my myself of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours yourself yourselves""".split())

_stopwords = _FALLBACK_STOP
_lemmatizer = None


def _init_nltk():
    """Try to use NLTK stopwords + WordNet lemmatizer; silently fall back if unavailable."""
    global _stopwords, _lemmatizer
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        sw = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # raises LookupError if wordnet is missing
        _stopwords, _lemmatizer = sw, lem
    except Exception:
        _stopwords, _lemmatizer = _FALLBACK_STOP, None


_init_nltk()
# Keep negations ("not good" must stay negative!) -> never treat them as stopwords.
STOPWORDS = {w for w in _stopwords if w not in NEGATIONS and w.replace("'", "") not in NEGATIONS}


def _simple_lemma(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def lemmatize(word: str) -> str:
    if _lemmatizer is not None:
        w = _lemmatizer.lemmatize(word, pos="v")
        return _lemmatizer.lemmatize(w)
    return _simple_lemma(word)


def tokenize(text: str):
    return text.split()


def clean_text(text) -> str:
    """Raw text -> cleaned, lemmatized string (same function for training AND live prediction)."""
    if not isinstance(text, str):
        return ""
    t = text.replace("’", "'")
    for emo, word in _EMOJI.items():
        t = t.replace(emo, word)
    t = t.lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)                       # keep the hashtag word
    t = t.replace("won't", "will not").replace("can't", "can not")
    t = re.sub(r"n't\b", " not", t)
    t = _NON_ALPHA.sub(" ", t)                       # punctuation, digits, remaining emojis
    t = _SPACES.sub(" ", t).strip()
    tokens = [w for w in tokenize(t) if w not in STOPWORDS and len(w) > 1]
    return " ".join(lemmatize(w) for w in tokens)
