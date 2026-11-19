# Move each file into its home

Use this table inside your **existing repository folder**. Keep its `.git`
directory in place: that directory holds the repository's history. The ZIP
supplies folders and the two attached Python/notebook files; it does not
contain the other documents visible only in the screenshot.

Windows hides the extensions in your screenshot. Enable **View → Show →
File name extensions** in File Explorer before renaming. Preserve the real
extension: renaming a file does not convert its format.

| Current visible name | Destination | Suggested new base name / action |
| --- | --- | --- |
| `CSCI 5535G_ Literature Review` | `references/` | `literature_review`; keep its spreadsheet extension |
| `dataminds_annotate` | `scripts/` | `annotate_posts.py` |
| `dataminds_codebook_v2_revised` | `data/annotations/` pending inspection | Keep the name initially; if it contains only label definitions, move it to `docs/codebook/` instead |
| `DataMinds_Full_Project_Process` | `docs/project/` | `project_process.pdf` |
| `DataMinds_Understanding Psychosocial F...` | `docs/project/` | `project_overview.pdf`; the original name is truncated in the screenshot |
| `dataminds_visualizations` | `reports/` | `visualizations.pdf` |
| `Easy version (4)` | `data/processed/` | `easy_dataset`; keep the actual CSV or Excel extension |
| `MINDMINER_Annotation_Codebook_v1` | `docs/codebook/` | `annotation_codebook_v1`; keep its Word extension |
| `MindMiner_CSCI5535G_Enhanced` | `reports/slides/` | `presentation`; keep its PowerPoint extension |
| `new_labels` | `data/annotations/` | `new_labels`; retain its extension until its role is confirmed |
| `train` | `data/splits/` | `train`; retain its extension |
| `val` | `data/splits/` | `val`; retain its extension |
| `test` | `data/splits/` | `test`; retain its extension |
| `Visualization_Guide` | `docs/` | `visualization_guide`; retain its Word extension |
| Attached `mindminer_utils (1).py` | `src/` | `mindminer_utils.py`; included in this package |
| Attached `notebook252d503acd.ipynb` | `notebooks/` | `02_transformer_classification.ipynb`; included with expanded explanations |

## What to use for the BERT experiment

Use the processed dataset with the `split` column. The transformer notebook
does not read `train`, `val`, and `test` as three separate files. Keep those
files as provenance until you establish how they relate to the processed
dataset. Do not rebuild partitions just to match the new folder names.

`data/annotations/` holds intermediate annotation exports;
`data/processed/` holds the chosen modeling input. Keep a clear record of
which file produced the reported results.

## Apply the structure without replacing the repository

1. Extract the ZIP somewhere separate so you can see its contents.
2. Copy the files and folders **inside** `mindminer/` into your existing
   project root. Review a filename conflict before replacing an existing file.
3. Move your screenshot files according to the table. Update any path used
   by an older annotation or visualization script before running that script.
4. Open the renamed transformer notebook and check the `DATA_FILE` setting.
5. Review the changed files in GitHub Desktop or with `git status`.
   Preview the diff before committing.

The package contains no `.git` directory and does not alter your GitHub
repository automatically. Existing dataset files already committed need
separate handling: ignore rules only affect untracked files. See the
[official GitHub instructions](https://docs.github.com/en/get-started/git-basics/ignoring-files).
