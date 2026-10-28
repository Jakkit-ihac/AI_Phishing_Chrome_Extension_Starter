from pathlib import Path
import pandas as pd

DATASET_PATH = Path("dataset/Dataset.csv")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("=" * 60)
    print("PHISHING DATASET INSPECTION")
    print("=" * 60)

    print(f"\nFile: {DATASET_PATH.resolve()}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    required = {"url", "label"}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    print("\n--- Missing Values ---")
    print(df[["url", "label"]].isna().sum().to_string())

    print("\n--- Label Distribution ---")
    print(df["label"].value_counts(dropna=False).to_string())

    print("\n--- Label Distribution (%) ---")
    percentages = (
        df["label"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )
    print(percentages.to_string())

    print("\n--- Duplicate URLs ---")
    urls = df["url"].astype("string").str.strip()

    print(f"Duplicate URL rows: {urls.duplicated().sum():,}")
    unique_urls = urls[urls.notna() & (urls != "")]
    print(f"Unique non-empty URLs: {unique_urls.nunique():,}")

    print("\n--- Labels Per URL ---")
    valid = df.dropna(subset=["url", "label"]).copy()
    valid["url_normalized"] = (
        valid["url"].astype("string").str.strip()
    )

    label_counts = valid.groupby("url_normalized")["label"].nunique()
    conflicts = label_counts[label_counts > 1]

    print(f"URLs with conflicting labels: {len(conflicts):,}")

    if len(conflicts):
        print("Examples of conflicting URLs:")
        print(conflicts.head(5).to_string())

    print("\n--- Sample URLs by Label ---")

    for label in sorted(df["label"].dropna().unique(), key=str):
        print(f"\nLabel = {label!r}")

        samples = df.loc[df["label"] == label, "url"].head(5)

        for url in samples:
            print(f"  {url}")

    print("\n--- Feature Column Types ---")
    print(df.dtypes.to_string())

    print("\nInspection complete. No data was modified.")


if __name__ == "__main__":
    main()
