import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATASET_PATH = PROJECT_ROOT / "dataset" / "cleaned_urls.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "splits"

sys.path.insert(0, str(BACKEND_DIR))


def get_hostname(url):
    value = str(url).strip()
    target = value if "://" in value else "//" + value

    try:
        hostname = urlsplit(target).hostname
        return hostname.lower().rstrip(".") if hostname else None
    except ValueError:
        return None


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df["hostname_group"] = df["url"].map(get_hostname)

    df = df.dropna(subset=["url", "label", "hostname_group"])
    df = df.reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels 0 and 1.")

    # First split: 80% development data, 20% test data.
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    development_idx, test_idx = next(
        first_split.split(
            df,
            y=df["label"],
            groups=df["hostname_group"],
        )
    )

    development = df.iloc[development_idx].copy()
    test = df.iloc[test_idx].copy()

    # Second split: 75% of development for train, 25% for validation.
    # Final proportions are approximately 60/20/20.
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    train_idx, validation_idx = next(
        second_split.split(
            development,
            y=development["label"],
            groups=development["hostname_group"],
        )
    )

    train = development.iloc[train_idx].copy()
    validation = development.iloc[validation_idx].copy()

    # Check that each split contains both classes.
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, part in splits.items():
        labels = set(part["label"].unique())

        if labels != {0, 1}:
            raise ValueError(
                f"{name} does not contain both labels: {labels}. "
                "We need to adjust the splitting strategy."
            )

    # Verify that the exact same hostname is not shared across splits.
    train_hosts = set(train["hostname_group"])
    validation_hosts = set(validation["hostname_group"])
    test_hosts = set(test["hostname_group"])

    overlaps = {
        "train_validation": train_hosts & validation_hosts,
        "train_test": train_hosts & test_hosts,
        "validation_test": validation_hosts & test_hosts,
    }

    for name, shared in overlaps.items():
        if shared:
            raise RuntimeError(
                f"Hostname leakage detected in {name}: {len(shared)}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        output = part[["url", "label"]].copy()
        output.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8",
        )

        print(f"\n--- {name.upper()} ---")
        print(f"Rows: {len(output):,}")
        print(f"Unique hostnames: {part['hostname_group'].nunique():,}")
        print("Label counts:")
        print(output["label"].value_counts().sort_index().to_string())
        print("Label percentages:")
        print(
            output["label"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
            .to_string()
        )

    print("\n--- Hostname Overlap Check ---")
    for name, shared in overlaps.items():
        print(f"{name}: {len(shared)} shared hostnames")

    print("\nSplit complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
