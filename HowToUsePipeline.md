# Pipline for Computational work
If your computer can't handle the coputational workload(like mines), you can use this pipeline to pass the workload to github.

---

## How it works

When you push code to `main`, GitHub Actions automatically:
1. Reads `FilesToRun.txt` to find which scripts to run
2. Runs each script in order
3. Uploads results to AWS S3
4. Commits figures back to this repo

You never need to touch AWS, the workflow file, or the UI. Just write your script, register it, and push.

---

### Step 1: Write your script in `src/`

Create a python file in the "src" folder. Name it after your task:

| Task | Filename |
|---|---|
| Exploratory Data Analysis | `EDA.py` |
| Citation Network | `Network.py` |
| Clustering | `Clustering.py` |
| Temporal Classification | `TemporalClassification.py` |

---

### Step 2: Follow the script template

Every script must follow this structure:

```python
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import os

#  1. Load the cleaned dataset from S3 
s3 = boto3.client("s3")
os.makedirs("cleaned", exist_ok=True)
s3.download_file(
  "data-science-citation-network",
  "cleaned/papers.parquet",
  "cleaned/papers.parquet"
)
df = pd.read_parquet("cleaned/papers.parquet")

# 2. Create your output folder 
# Use the folder that matches your task (only one):
os.makedirs("outputs/figures/eda", exist_ok=True)
os.makedirs("outputs/figures/network", exist_ok=True)
os.makedirs("outputs/figures/clustering", exist_ok=True)
os.makedirs("outputs/figures/classification", exist_ok=True)

# 3. Do your analysis 

# ... YOU CODE GOES HERE ...

#  4. Save every figure like this 
plt.savefig("outputs/figures/eda/figure_name.png", dpi=150, bbox_inches="tight")
plt.close()  # always close after saving
```

---

### Step 3: Add your script to `FilesToRun.txt`

Open `FilesToRun.txt` in the root of the repo and add a line for your script:

FORMAT: <FILE_NAME> | <RESULTS_DESTINATION>
```
eda.py | outputs/
```

Format is: `script_name.py | outputs/`

The second part tells the pipeline where to upload your results. Always use `outputs/`.

Example `FilesToRun.txt`:
```
preprocess.py | outputs/
eda.py | outputs/
network.py | outputs/
clustering.py | outputs/
classification.py | outputs/
```
**Make sure to delete file name/destination form FilesToRun.txt**
---

### Step 4: Push to main
* Use what ever method to push

GitHub Actions will pick it up automatically. You can watch it run under the **Actions** tab on GitHub.

---

### Step 5: Check your results

Once the pipeline finishes (~10 minutes), check:
- **GitHub repo**: `outputs/figures/your-task/` for the raw image files
- **UI**: https://sal-gr.github.io/Data-Science-1 for the visual dashboard

---

## **Rules**

### Files and folders
- Write all scripts in `src/`
- Save all figures to `outputs/figures/your_task/`
- Save figures as `.png`
- Never save files outside the `outputs/` folder

### Plotting
- **Never use `plt.show()`**, it will cause the pipeline to hang indefinitely
- Always use `plt.savefig()` followed by `plt.close()`
- Use `dpi=150` and `bbox_inches="tight"` for figures

### Data access
- Always read the cleaned dataset from S3 using the template above
- Never hardcode local file paths, the pipeline runner uses a fresh environment every time
- Never modify the raw or cleaned dataset

### Files you must never edit
| File | Reason |
|---|---|
| `.github/workflows/pipeline.yml` | Controls the entire automation |
| `src/runner.py` | Manages script execution and uploads |
| `src/preprocess.py` | Dataset is already cleaned and stored in S3 |
| `docs/index.html` | The results UI |
| `requirements.txt` | Ask before adding new dependencies |

### FilesToRun.txt
- Always add your script before pushing
- Scripts not listed here will not run
- remove script when finished

---

## Adding new dependencies

If your script needs a library that isn't in `requirements.txt`, add it

---

## Dataset reference

The cleaned dataset is a pandas DataFrame with these columns:

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique paper ID |
| `title` | string | Paper title |
| `authors` | list | List of author names |
| `venue` | string | Conference or journal (may be empty) |
| `year` | int | Publication year |
| `n_citation` | int | Citation count |
| `references` | list | List of cited paper IDs |
| `abstract` | string | Paper abstract (may be empty) |
| `author_count` | int | Number of authors |
| `reference_count` | int | Number of references |
| `text_combined` | string | Title + abstract concatenated |

---