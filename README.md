# MindMiner
### Learning psychosocial labels from text

MindMiner is a data mining project that predicts **stressors, relationships,
emotions, and responses** from a post. A post can receive several labels at
once, so the task is **multilabel text classification**.

For example, the fictional sentence “My exams are overwhelming, and I would
appreciate some advice” might receive `study`, `anxiety`, and `advice seeking`.
These example labels illustrate the task; actual annotation decisions follow
the project codebook.

**Start here:** [Your first walkthrough](docs/START_HERE.md) ·
[BERT explained from the beginning](docs/BERT_FROM_BASICS.md) ·
[Where your existing files belong](docs/FILE_MAP.md)

## What is included

| Component | Status |
| --- | --- |
| BERT, RoBERTa, and DistilBERT classification | Notebook included |
| Shared loading, label encoding, folds, and evaluation | Python helper included |
| Annotation script, original documents, slides, and literature spreadsheet | Placement guide provided; original files need to be added |
| Dataset and model results | Not bundled; add your approved local dataset and generate results |
| Traditional-model and prompted-LLM experiments | Reserved places in the workflow; code not supplied in this package |

## Find your way around

| Location | What belongs here | Simple way to think about it |
| --- | --- | --- |
| `notebooks/` | Step-by-step experiments | The lab workbook |
| `src/` | Reusable Python functions | The shared toolbox |
| `scripts/` | Standalone annotation or preparation programs | Programs run as individual jobs |
| `data/` | Local datasets and annotation exports | The project inputs |
| `docs/` | Explanations, codebook, and project plan | The instruction manual |
| `references/` | Literature-review spreadsheet and reading notes | The reading shelf |
| `results/` | Generated experiment files, kept local | The experiment's output folder |
| `reports/` | Reviewed figures, tables, slides, and reports | The material you present |
| `models/` | Optional saved model weights, kept local | The trained models |

## Run the transformer experiment

1. Create a Python environment. Install the appropriate PyTorch build for
   your computer using the [official installation selector](https://pytorch.org/get-started/locally/).
2. From the project root, install the remaining packages and open JupyterLab:

   ```bash
   python -m pip install -r requirements.txt
   python -m jupyterlab
   ```

3. Put the approved dataset at `data/processed/easy_dataset.csv`.
   For Excel, change `DATA_FILE` in the notebook to the actual Excel path.
4. Open [02_transformer_classification.ipynb](notebooks/02_transformer_classification.ipynb).
   Read the explanations and run the cells from top to bottom.

The notebook finds `src/mindminer_utils.py` when opened from the root or
`notebooks/`. The helper must have that exact name. For Kaggle, see
[the setup walkthrough](docs/START_HERE.md#running-on-kaggle).

**A full run is substantial:** the supplied settings train three models
across five folds, then refit each model once: **18 training runs**, each
with **10 epochs**. No training or dataset-specific evaluation has been
performed while preparing this package.

## What the models see

| Column(s) | Purpose |
| --- | --- |
| `text_clean` | The only prediction input |
| `stressors`, `relationships`, `emotions`, `responses` | Reference labels to learn and evaluate |
| `belong`, `burden`, `group` | TBe/PBu metadata for fold balancing and subgroup reporting |
| `post_id`, `orig_row_id`, `split` | Identity, provenance, and partition metadata |

The helper declares **30 labels: 10 stressors + 7 relationships + 8 emotions
+ 5 responses**. This is the vocabulary in the supplied code; confirm it
matches the current dataset and codebook. The dataset was not supplied here.
[Read the schema and exact labels](data/README.md).

## How evaluation works

The existing `train` and `val` rows form the development set. Five-fold
cross-validation trains a fresh model on four folds and evaluates it on the
fifth. The highest mean development macro F1 selects the model before test
scores are inspected. Each predeclared model is then refit on all development
rows and evaluated once on the separate test partition for comparison.

The supplied experiment uses fixed hyperparameters and a fixed 0.5 decision
threshold. It uses **five-fold cross-validation, without an inner tuning
loop**. Call it nested cross-validation only if an inner tuning loop is
actually implemented.

[Understand every output and metric](docs/RESULTS_GUIDE.md).

## Sharing this project

The supplied ignore rules keep datasets, row-level predictions, model weights,
caches, and raw experiment outputs out of new commits. Share reviewed aggregate
figures and tables through `reports/`. Clear notebook outputs before sharing a
notebook that has been run on real posts.

A `.gitignore` rule does not remove files already tracked by Git or erase
earlier commits. See [GitHub's guide to ignoring files](https://docs.github.com/en/get-started/git-basics/ignoring-files).

## Reproducing a completed run

The experiment records its dataset hash, available package versions, seeds,
label list, settings, fold assignments, and resolved model commits. Keep these
alongside the results. After a successful run, record the complete environment:

```bash
python -m pip freeze > results/environment.txt
```

The supplied `requirements.txt` lists dependencies rather than a tested,
fully locked environment. Seeds aid repeatability; hardware and library
differences can still affect results. Report actual run settings and scores.
