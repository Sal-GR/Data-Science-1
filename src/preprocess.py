import pandas as pd
import json
import boto3
import os

def load_raw_data():
    files = ["raw/dblp-ref-0.json", "raw/dblp-ref-1.json", "raw/dblp-ref-2.json", "raw/dblp-ref-3.json"]
    records = []
    for f in files:
        print(f"Reading {f}...")
        with open(f, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(records)

def clean_data(df):
    print("Cleaning data...")

    df = df.dropna(subset=["id", "title", "year"])
    df = df[df["title"].str.strip() != ""]
    df = df.drop_duplicates(subset=["id"])
    df["abstract"] = df["abstract"].fillna("")
    df["venue"] = df["venue"].fillna("")
    df["author_count"] = df["authors"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["reference_count"] = df["references"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["text_combined"] = df["title"] + " " + df["abstract"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    print(f"Records after cleaning: {len(df)}")
    return df

def download_from_s3():
    print("Downloading raw data from S3...")

    s3 = boto3.client("s3")
    bucket = "data-science-citation-network"
    os.makedirs("raw", exist_ok=True)

    for i in range(4):
        filename = f"dblp-ref-{i}.json"
        print(f"Downloading {filename}...")
        s3.download_file(bucket, f"raw/{filename}", f"raw/{filename}")

def upload_to_s3(filepath):
    print("Uploading cleaned data to S3...")

    s3 = boto3.client("s3")
    s3.upload_file(filepath, "data-science-citation-network", "cleaned/papers.parquet")
    print("Upload complete!")

if __name__ == "__main__":
    download_from_s3()
    df = load_raw_data()
    df = clean_data(df)

    os.makedirs("cleaned", exist_ok=True)
    df.to_parquet("cleaned/papers.parquet", index=False)
    upload_to_s3("cleaned/papers.parquet")
    
    print(f"Done! Final shape: {df.shape}")