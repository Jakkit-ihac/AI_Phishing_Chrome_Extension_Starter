import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit


SUSPICIOUS_KEYWORDS = (
    "login",
    "signin",
    "verify",
    "verification",
    "password",
    "account",
    "secure",
    "update",
    "confirm",
    "wallet",
    "webscr",
    "ebayisapi",
    "banking",
    "authenticate",
    "credential",
    "recover",
    "suspend",
    "unlock",
    "payment",
)


def _entropy(value: str) -> float:
    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def extract_features(url: str) -> dict[str, float]:
    """
    Extract numeric features from a URL without requesting the target website.

    The returned feature names and order must remain consistent between
    model training and API inference.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()

    if len(url) > 8192:
        raise ValueError("URL exceeds the maximum allowed length")

    # Parse URLs with or without a scheme.
    parse_target = url if "://" in url else "//" + url

    try:
        parsed = urlsplit(parse_target)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""
        scheme = parsed.scheme.lower()
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL format") from exc

    if not hostname:
        raise ValueError("URL must contain a hostname")

    # Avoid treating a user-info field as part of the hostname.
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        is_ip = 1
    except ValueError:
        is_ip = 0

    url_lower = url.lower()
    hostname_parts = hostname.split(".")
    letters = sum(char.isalpha() for char in url)
    digits = sum(char.isdigit() for char in url)
    special = sum(not char.isalnum() for char in url)

    keyword_count = sum(
        1 for keyword in SUSPICIOUS_KEYWORDS
        if keyword in url_lower
    )

    # This is a heuristic count, not a public-suffix-aware domain parser.
    subdomain_count = max(0, len(hostname_parts) - 2)

    features = {
        "url_length": float(len(url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "fragment_length": float(len(fragment)),
        "subdomain_count": float(subdomain_count),
        "is_ip": float(is_ip),
        "has_https": float(scheme == "https"),
        "has_http": float(scheme == "http"),
        "has_port": float(port is not None),
        "has_at_symbol": float("@" in url),
        "has_punycode": float("xn--" in hostname),
        "has_percent_encoding": float("%" in url),
        "has_suspicious_keyword": float(keyword_count > 0),
        "suspicious_keyword_count": float(keyword_count),
        "dot_count": float(url.count(".")),
        "dash_count": float(url.count("-")),
        "underscore_count": float(url.count("_")),
        "slash_count": float(url.count("/")),
        "question_count": float(url.count("?")),
        "equals_count": float(url.count("=")),
        "ampersand_count": float(url.count("&")),
        "percent_count": float(url.count("%")),
        "digit_count": float(digits),
        "letter_count": float(letters),
        "special_count": float(special),
        "digit_ratio": _safe_ratio(digits, len(url)),
        "special_ratio": _safe_ratio(special, len(url)),
        "hostname_entropy": _entropy(hostname),
    }

    return features
