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

def generate_metadata(df):
    import json
    print("Generating metadata...")
    os.makedirs("outputs/data", exist_ok=True)

    summary = {
        "total_papers": int(len(df)),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "unique_venues": int(df["venue"].nunique()),
        "missing_abstracts": int((df["abstract"] == "").sum()),
        "missing_venue": int((df["venue"] == "").sum()),
        "avg_citations": round(float(df["n_citation"].mean()), 2),
        "avg_authors": round(float(df["author_count"].mean()), 2),
        "avg_references": round(float(df["reference_count"].mean()), 2),
        "total_citations": int(df["n_citation"].sum()),
    }
    with open("outputs/data/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary.json")

    sample = df[["id", "title", "authors", "venue", "year", "n_citation"]].head(100)
    sample["authors"] = sample["authors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    sample.to_json("outputs/data/sample.json", orient="records", indent=2)
    print("Saved sample.json")

    s3 = boto3.client("s3")
    s3.upload_file("outputs/data/summary.json", BUCKET, "outputs/data/summary.json")
    s3.upload_file("outputs/data/sample.json", BUCKET, "outputs/data/sample.json")
    print("Uploaded metadata to S3")

if __name__ == "__main__":
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=BUCKET, Key="cleaned/papers.parquet")
        print("Cleaned data already exists in S3. Skipping preprocessing.")
        
        print("Downloading cleaned data to generate metadata...")
        os.makedirs("cleaned", exist_ok=True)
        print("Loading cleaned data for metadata generation...")
        s3.download_file(BUCKET, "cleaned/papers.parquet", "cleaned/papers.parquet")
        df = pd.read_parquet("cleaned/papers.parquet")
        generate_metadata(df)
        exit(0)
    except Exception as e:
        print(f"Starting preprocessing... ({e})")

    download_from_s3()
    output_path = process_files()
    upload_to_s3(output_path)
    generate_metadata(pd.read_parquet(output_path))
    print("\nPreprocessing complete!")
