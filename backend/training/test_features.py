from app.ml.features import extract_features

test_urls = [
    "https://example.com/",
    "http://login.example.test/account?verify=1",
    "https://192.0.2.10/login",
    "https://shop.example.test/products/item-1",
]

for url in test_urls:
    print("\nURL:", url)

    try:
        features = extract_features(url)
        print("Feature count:", len(features))
        print("Hostname length:", features["hostname_length"])
        print("Is IP address:", features["is_ip"])
        print("HTTPS:", features["has_https"])
        print("Suspicious keyword count:",
              features["suspicious_keyword_count"])
    except ValueError as exc:
        print("Validation error:", exc)

