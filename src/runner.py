import subprocess
import boto3
import os
import sys

BUCKET = "data-science-citation-network"

def parse_files_to_run():
    entries = []
    with open("FilesToRun.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2:
                script, s3_destination = parts
                entries.append((script, s3_destination))
            else:
                print(f"Skipping malformed line: {line}")
    return entries

def upload_outputs(local_dir, s3_destination):
    if not os.path.exists(local_dir):
        print(f"  No output folder found at {local_dir}, skipping upload.")
        return
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = s3_destination + relative_path.replace("\\", "/")
            print(f"  Uploading {local_path} → s3://{BUCKET}/{s3_key}")
            s3.upload_file(local_path, BUCKET, s3_key)

def run_script(script, s3_destination):
    script_path = os.path.join("src", script)
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found. Skipping.")
        return False

    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print(f"Results will upload to: s3://{BUCKET}/{s3_destination}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        return False

    print(f"\nUploading results for {script}...")
    upload_outputs("outputs", s3_destination)
    return True

if __name__ == "__main__":
    entries = parse_files_to_run()
    if not entries:
        print("FilesToRun.txt is empty or has no valid entries.")
        sys.exit(0)

    print(f"Found {len(entries)} script(s) to run:")
    for script, dest in entries:
        print(f"  {script} → {dest}")

    failed = []
    for script, s3_destination in entries:
        success = run_script(script, s3_destination)
        if not success:
            failed.append(script)

    if failed:
        print(f"\nThe following scripts failed: {failed}")
        sys.exit(1)
    else:
        print("\nAll scripts completed successfully!")