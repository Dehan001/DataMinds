# Your first walkthrough

Think of the project as a small research lab. The dataset is the material
you study, the notebook records the experiment, and the report explains what
you found. The folders help a new reader follow that same story.

## 1. Begin with the question

We want to read a post and predict the kinds of stressors, relationships,
emotions, and responses it contains. Several answers can be correct for the
same post. This is why we create one yes/no output for each possible tag.

TBe means **thwarted belongingness**; PBu means **perceived burdensomeness**.
This notebook preserves their annotations for subgroup analysis and fold
balancing. Its prediction targets are the four tag families, not TBe/PBu.

## 2. Know the three files you will use first

| File | Why you need it |
| --- | --- |
| `requirements.txt` | Lists packages the experiment imports; PyTorch is installed separately |
| `src/mindminer_utils.py` | Loads the data, makes label vectors/folds, and calculates metrics |
| `notebooks/02_transformer_classification.ipynb` | Explains and runs the experiment in small steps |

A **function** is a named piece of reusable code. The notebook calls helper
functions such as `load_data()` instead of repeating the whole procedure.
An **import** makes those functions available in the notebook.

## 3. Add the dataset

Place your approved file at `data/processed/easy_dataset.csv`. If it is
Excel, preserve its extension and update `DATA_FILE` in the settings cell.
Read [the schema](../data/README.md) before running. Label spelling must
match the declared vocabulary, and multi-tag cells use semicolons.

Renaming a spreadsheet to `.csv` does not convert it. Use Excel's export
function if you intentionally want a CSV conversion.

## 4. Prepare Python and open the notebook

Use an isolated Python environment. Select and install PyTorch for your
hardware using its [official installer guide](https://pytorch.org/get-started/locally/).
Then, from the project root, run:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Choose a notebook kernel that uses this environment. Open the transformer
notebook. A notebook contains **Markdown cells** for explanations and
**code cells** that execute. Run a cell using Shift+Enter, then read its
output before continuing.

## 5. Read the experiment as a story

| Stage | The question it answers |
| --- | --- |
| Task and toy example | What are we asking the model to predict? |
| Settings | Which models and training choices will we use? |
| Data preparation | Are rows, labels, and partitions usable? |
| Tokenization | How does a sentence become numbers? |
| Training | How does the model learn from annotated posts? |
| Cross-validation | Does learning transfer to unseen development posts? |
| Final test | How well does the fixed experiment work on the separate test set? |
| Interpretation | Which labels and groups need attention? |

Start with explanations and the small teaching examples. Full training runs
18 separate fits with the current settings. A CPU is supported by the code,
but transformer training can take much longer there.

## Running on Kaggle

Upload `02_transformer_classification.ipynb` as a notebook. Add your dataset
as input. Upload the helper using the exact filename `mindminer_utils.py`.
The notebook looks for that helper in `/kaggle/working` first, then in the
original helper-input folder:

```text
/kaggle/input/datasets/farhannoordehan02629/dataminds-py
```

If Kaggle assigns a different helper-input directory, set
`MINDMINER_HELPER_DIR` at the top of the setup cell to that directory. Only
one matching helper file should be present there. The original data path is
retained as a fallback when it exists:

```text
/kaggle/input/datasets/farhannoordehan02629/dataminds/Easy version (4).csv
```

Change `DATA_FILE` if the mounted dataset path differs. Kaggle results are
written under `/kaggle/working/results/transformers`. Use the notebook's
optional package-install cell if the kernel lacks dependencies. PyTorch
must already be available in that kernel. Allow model downloads when running
for the first time, or provide the pretrained models through a local cache.

## 6. Share a readable project

Place reviewed charts and tables in `reports/`, and explain their meaning.
Keep local datasets and generated row-level predictions in the ignored
folders. The ZIP already has empty notebook outputs. Before sharing a later
executed version, clear its outputs and review what will be committed.

Keep your existing `.git` directory in the project root. Review the
[file-by-file moving guide](FILE_MAP.md), then inspect the Git diff and
commit the changes you intend to share.
