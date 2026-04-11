import pandas as pd
import json
import boto3
import os
import gc

BUCKET = "data-science-citation-network"
CHUNK_SIZE = 50000

def download_from_s3():
    print("Downloading raw data from S3...")
    s3 = boto3.client("s3")
    os.makedirs("raw", exist_ok=True)
    for i in range(4):
        filename = f"dblp-ref-{i}.json"
        local_path = f"raw/{filename}"
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            s3.download_file(BUCKET, f"raw/{filename}", local_path)
        else:
            print(f"Already exists, skipping: {filename}")

def clean_chunk(records):
    df = pd.DataFrame(records)
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
    return df

def process_files():
    os.makedirs("cleaned", exist_ok=True)
    output_path = "cleaned/papers.parquet"
    files = [f"raw/dblp-ref-{i}.json" for i in range(4)]
    
    all_chunks = []
    total_records = 0
    chunk_count = 0

    for filepath in files:
        print(f"\nProcessing {filepath}...")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

                if len(records) >= CHUNK_SIZE:
                    chunk_df = clean_chunk(records)
                    all_chunks.append(chunk_df)
                    total_records += len(chunk_df)
                    chunk_count += 1
                    print(f"  Chunk {chunk_count}: {len(chunk_df)} records (total so far: {total_records})")
                    records = []
                    gc.collect()

            if records:
                chunk_df = clean_chunk(records)
                all_chunks.append(chunk_df)
                total_records += len(chunk_df)
                chunk_count += 1
                print(f"  Chunk {chunk_count}: {len(chunk_df)} records (total so far: {total_records})")
                records = []
                gc.collect()

    print(f"\nCombining {len(all_chunks)} chunks...")
    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()

    print("Removing duplicates across files...")
    df = df.drop_duplicates(subset=["id"])
    print(f"Final record count: {len(df)}")

    print("Saving to parquet...")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path

def upload_to_s3(filepath):
    print("\nUploading cleaned data to S3...")
    s3 = boto3.client("s3")
    s3.upload_file(filepath, BUCKET, "cleaned/papers.parquet")
    print("Upload complete!")

if __name__ == "__main__":
    download_from_s3()
    output_path = process_files()
    upload_to_s3(output_path)
    print("\nPreprocessing complete!")
    