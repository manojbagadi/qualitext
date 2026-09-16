"""
Feature Extraction Module for Sentence Quality Classification
Extracts 9 structural and lexical quality indicators from raw text.
"""
import re, string

SPAM_TERMS = {
    "free", "win", "claim", "cash", "prize", "urgent", "lottery", "bonus",
    "casino", "credit", "loan", "bitcoin", "viagra"
}

def extract_features(text: str):
    t = str(text).strip()
    words = t.split() or [""]
    chars = max(len(t), 1)
    letters = [c for c in t if c.isalpha()] or ["a"]
    
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    rep_punct = 1.0 if re.search(r'[!?$*#%]{2,}', t) else 0.0
    dollar_count = t.count('$') / chars
    excess_exclam = max(0, t.count('!') - 1) / chars
    cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    digit_density = sum(1 for c in t if c.isdigit()) / chars
    vowel_ratio = sum(1 for c in t.lower() if c in 'aeiou') / len(letters)
    spam_hits = float(sum(1 for w in [w.strip(string.punctuation).lower() for w in words] if w in SPAM_TERMS))
    valid_ratio = sum(1 for w in words if len(w) >= 2 and any(c in 'aeiou' for c in w.lower())) / max(len(words), 1)
    
    return [avg_len, rep_punct, dollar_count, excess_exclam, cap_ratio, digit_density, vowel_ratio, spam_hits, valid_ratio]
