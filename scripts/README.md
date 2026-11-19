# Standalone programs

## `annotate_posts.py`

`annotate_posts.py` labels social media posts using a locally running Ollama
model and the MINDMINER codebook. The default model is **`qwen3.5:27b`**.

Here, **annotation** means assigning descriptive tags to a post. The script
asks the model to select tags and provide short, exact quotes as evidence.
It uses a pretrained model for annotation; it does not train a classifier.

The codebook contains **31 tags** across four independent fields:

| Field | What it describes | Tags |
| --- | --- | --- |
| Stressors | The difficulties discussed in the post | 10 |
| Relationships | The people or groups involved | 7 |
| Emotions | The writer's expressed feelings | 8 |
| Responses | The writer's help-seeking or coping responses | 6 |

A post can receive several tags. For example, a fictional post about feeling
anxious after a family argument and requesting advice may support `family
conflict`, `family`, `anxiety`, and `advice seeking`. Each field is evaluated
separately, and every selected tag needs textual evidence.

## File placement

Use these locations in the project:

| Location | Purpose |
| --- | --- |
| `scripts/annotate_posts.py` | The annotation program |
| `scripts/README.md` | This guide |
| `data/splits/train.csv` | Original training posts |
| `data/splits/val.csv` | Original validation posts |
| `data/splits/test.csv` | Original test posts |
| `data/annotations/new_labels.csv` | Generated annotations |
| `data/annotations/new_labels.meta.json` | Settings and run history |

If the downloaded program is named `annotate_posts.py.py`, rename it to
`annotate_posts.py` before using the commands below. Keep one `.py` extension.

## 1. Prepare the environment

Use **Python 3.10 or later**. On Windows, you can use Anaconda Prompt with
your Python environment activated. Install the script's Python dependency:

```bash
python -m pip install tqdm
```

Install [Ollama](https://docs.ollama.com/quickstart), open its application or
start its local server, and download the model used by the script:

```bash
ollama pull qwen3.5:27b
```

If the Ollama server is not already running, start it in a separate terminal:

```bash
ollama serve
```

The program connects to `http://127.0.0.1:11434/api/`. Ollama handles model
execution; the Python script communicates with it using Python's standard
HTTP library.

## 2. Prepare the input CSVs

Every input CSV must contain a column named **`text`** with the post content.
The header must match exactly. The script does not use `text_clean` as a
replacement for `text`.

An optional `post_id` column provides a stable identifier. If several possible
ID columns exist, the script checks them in this order: `Unnamed: 0`,
`post_id`, then `id`. When none exists, it assigns row numbers starting at 1
within each file. IDs must be nonempty and unique within each input file.

When a folder is supplied, the script reads its immediate `*.csv` files.
It does not search nested folders. The filename stem becomes the output
`split`: for example, `train.csv` produces `train`. An existing `split`
column in the input does not override that filename-based value.

Only the post's `text` is sent to the model. Existing TBe/PBu annotations,
such as `belong` and `burden`, are not included in the model input.

## 3. Start with a small run

Run this command from the **project root**, the folder containing `scripts/`
and `data/`. It processes up to five unfinished posts:

```bash
python scripts/annotate_posts.py --data-dir data/splits --output data/annotations/new_labels.csv --limit 5
```

Inspect the selected tags, evidence, and review notes. Then process the remaining
posts using the same output file:

```bash
python scripts/annotate_posts.py --data-dir data/splits --output data/annotations/new_labels.csv
```

To read just one CSV, pass its path to `--data-dir` instead of a folder.

The explicit paths above matter: without them, the program looks for `data/`
beside the script and writes `new_labels.csv` beside the script. With the
recommended placement, those defaults would be `scripts/data/` and
`scripts/new_labels.csv`.

## 4. Understand how annotation works

1. The script reads a post and sends it with the codebook to Ollama.
2. For every allowed tag, the model returns an exact supporting quote or
   `null` when the tag is unsupported.
3. The script checks the JSON structure, allowed tag names, value types,
   and whether every selected quote appears exactly in the original text.
4. Valid selections become semicolon-separated tags in the output CSV.
   If a field has no supported tags, it is written as **`not stated`**.
5. The script saves progress after every processed post.

`not stated` is an empty-field marker, not a 32nd tag. Every field has an
`other` tag for explicit evidence outside its named categories. Selecting
`other` requires an explanation in `review_note`; it is not a way to express
uncertainty. Genuine ambiguity uses `needs_review=true` with an explanation.

The checks verify structure and quoted evidence. They do not establish that
the model interpreted the evidence correctly. The annotations remain
model-generated labels for human review.

## 5. Read the output

The CSV includes these columns:

| Column(s) | Meaning |
| --- | --- |
| `post_id`, `split` | The post identifier and input filename stem |
| `stressors`, `relationships`, `emotions`, `responses` | Selected tags, separated by semicolons, or `not stated` |
| `needs_review` | Whether human judgment is requested |
| `review_note` | Explanation of ambiguity, an `other` selection, or a processing failure |
| `evidence` | JSON containing selected tags and their supporting quotes; failed rows may contain the invalid model response |
| `status` | Processing state, explained below |
| `text_sha256` | A hash used to detect changes to the input text |
| `annotated_at` | Annotation timestamp in UTC |

| Status | Meaning | What happens on the next run? |
| --- | --- | --- |
| `pending` | The post has not been processed | It is eligible for processing |
| `labeled` | Valid model output with no review flag | It is skipped |
| `review` | Valid model output requesting human judgment | It is skipped; review it separately |
| `failed` | No valid annotation was produced | It is retried |

An `other` selection requires a note but does not automatically set the
status to `review`. That status follows the model's `needs_review` value.
Blank fields in a failed row do not mean that the post has no tags.

The `.meta.json` file records the model name and digest, script version,
prompt/schema hashes, generation settings, and run history. Keep it with the
CSV so that the program can check whether a resumed run is compatible.

## 6. Stop, resume, and retry

Press **Ctrl+C** to stop. Completed rows have already been saved. Rerun the
command with the same input and output paths to continue unfinished work.
Keep the metadata file alongside the CSV.

The default is **five attempts per post**. Invalid responses and request
failures are retried with increasing delays. Set `--max-attempts 0` to keep
retrying without an attempt limit. Empty text and inputs that exceed the
script's conservative context budget are recorded as failed for review
instead of being retried indefinitely within that run.

Changes to saved post text or incompatible model, prompt, schema, or context
settings require a new output filename. This keeps different annotation runs
from being mixed together.

## Command-line options

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | `data/` beside the script | An input CSV folder or one CSV file |
| `--output` | `new_labels.csv` beside the script | Destination CSV, separate from the inputs |
| `--model` | `qwen3.5:27b` | Exact name of an installed Ollama model |
| `--limit` | `0` | Maximum unfinished posts to process; 0 means all |
| `--context` | `32768` | Requested context size; minimum 8192 |
| `--timeout` | `900` | Seconds allowed for each annotation request |
| `--max-attempts` | `5` | Attempts per post; 0 means unlimited |

The generation settings are temperature `0`, seed `42`, maximum generated
tokens `4096`, and `think=False`. They are recorded in the metadata.

## Before using the labels for classification

Review ambiguous and failed rows before treating the output as a modeling
dataset. Join the annotations back to the original posts using `split` and
`post_id` when you need text and other metadata; these are not copied into
the annotation output automatically.

For the earlier classification helper, convert `not stated` to an empty tag
cell and use the updated **31-label** vocabulary, including `responses::other`.
The annotation CSV is an intermediate output, not the complete 11-column
classification dataset.

Evidence contains excerpts from the original posts. Keep annotation outputs
in the project's ignored data folder and share reviewed aggregate results.

## Common problems

| Message or symptom | What to check |
| --- | --- |
| `needs a text column` | The CSV header must contain `text`; this script does not substitute `text_clean` |
| `No CSV files found` | Check `--data-dir`; folder loading is not recursive |
| `Model is not installed` | Download the exact model named by `--model` |
| Connection error | Check that Ollama is running on the local endpoint above |
| Context-budget failure | Check post length and use a supported larger `--context` with a new output filename |
| Model/settings changed | Use a new output filename for the changed experiment |
| Output metadata is missing | Restore the matching `.meta.json` file or start with a new output filename |
| A `review` row is not rerun | This status is considered processed; review it separately |
