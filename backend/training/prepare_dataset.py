from pathlib import Path

import pandas as pd


DATASET_PATH = Path("dataset/Dataset.csv")
OUTPUT_PATH = Path("dataset/cleaned_urls.csv")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # ใช้เฉพาะ URL และ Label จากไฟล์ต้นฉบับ
    # จะสร้าง Features จาก URL เองในขั้นตอนถัดไป
    df = df[["url", "label"]].copy()

    # ทำความสะอาดข้อมูลพื้นฐาน
    df["url"] = df["url"].astype("string").str.strip()
    df = df.dropna(subset=["url", "label"])
    df = df[df["url"] != ""]

    # ตรวจสอบว่า Label เป็นตัวเลขและมีค่า 0 หรือ 1 เท่านั้น
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    invalid_labels = ~df["label"].isin([0, 1])
    if invalid_labels.any():
        raise ValueError(
            "พบ Label ที่ไม่ใช่ 0 หรือ 1: "
            f"{df.loc[invalid_labels, 'label'].unique().tolist()}"
        )

    df["label"] = df["label"].astype("int64")

    # URL เดียวกันต้องมี Label เดียวกัน
    label_counts = df.groupby("url")["label"].nunique()
    conflicts = label_counts[label_counts > 1]

    if len(conflicts):
        raise ValueError(
            f"พบ URL ที่มี Label ขัดแย้งกัน {len(conflicts)} รายการ"
        )

    before = len(df)

    # ลบ URL ซ้ำ โดยไม่เปลี่ยน Label
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nPreparation complete")
    print(f"Original rows: {before:,}")
    print(f"Rows after deduplication: {len(df):,}")
    print(f"Duplicate rows removed: {before - len(df):,}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nLabel percentages:")
    print(
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .to_string()
    )
    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
