# Zero-shot LLM classification

## `evaluate_ollama_llms.py`

This script compares three local language models on the MINDMINER multilabel
classification task. Each model reads a post and predicts its **stressors,
relationships, emotions, and responses** using the shared codebook.

The current codebook has **31 tags: 10 stressors, 7 relationships, 8 emotions,
and 6 responses**. Each field includes `other`. One post can receive several
tags, and an empty list means no tag is supported in that field.

## What does “zero-shot” mean?

The model receives instructions, tag definitions, and the post, without
labeled demonstration examples. It uses its existing language capabilities
to return the requested tags. The script does not fine-tune model weights.

For a fictional post such as “I feel anxious about my exams. Can someone offer
advice?”, plausible tags might include `study`, `anxiety`, and `advice seeking`.
Actual decisions must follow the codebook boundaries and the post's evidence.

The prediction is then compared with the reference annotations already stored
in the dataset. This makes the script an **evaluation program**. Annotation
review notes and evidence quotes are outside this benchmark's four-array output
schema, even where the shared definitions describe annotation-workflow notes.

| Program | Purpose | Model output |
| --- | --- | --- |
| `annotate_posts.py` | Create annotation records for review | Tags, exact evidence quotes, and review notes |
| `evaluate_ollama_llms.py` | Evaluate zero-shot predictions against an existing reference | Four lists of predicted tags |

Qwen is excluded from this benchmark's model list. If the reference labels
were originally generated using Qwen, the scores describe agreement with that
reference; independent human evaluation is needed to establish human-label
performance. Excluding Qwen does not remove all possible reference-label bias.

## Models

| Model | Ollama name used by the script |
| --- | --- |
| Mistral | `mistral:7b-instruct` |
| Gemma | `gemma:2b` |
| Llama | `llama3.2:3b` |

All three receive the same fixed instructions, codebook, output schema, and
selected posts. Their predictions are scored using the same metric functions.

## Project files

| Path | Role |
| --- | --- |
| `scripts/evaluate_ollama_llms.py` | The evaluation program |
| `src/mindminer_codebook.py` | Shared tag names and definitions |
| `src/mindminer_utils.py` | Data preparation, label encoding, and metrics |
| `data/processed/easy_dataset.csv` | Default processed input |
| `results/ollama/` | Generated predictions, caches, settings, and reports |

Keep the complete repository together so the script can import the shared
modules. It finds `src/` from its own location rather than requiring helpers
beside the script.

## 1. Install the requirements

Use Python 3.10 or later. From the project root, install the project packages:

```bash
python -m pip install -r requirements.txt
```

The benchmark uses the `ollama` Python package, NumPy, pandas, scikit-learn,
and the shared utilities. The optional development-subset report also uses
`iterative-stratification`. Excel input requires its matching reader engine.

Install and open [Ollama](https://docs.ollama.com/quickstart). Download the
configured models:

```bash
ollama pull mistral:7b-instruct
ollama pull gemma:2b
ollama pull llama3.2:3b
```

If the local server is not already running, start it in a separate terminal:

```bash
ollama serve
```

The script connects to `http://localhost:11434` by default. Run it where that
endpoint can reach your Ollama service. A hosted notebook's `localhost` refers
to the hosted environment, not your personal computer.

## 2. Prepare the dataset

Use the same processed schema as the traditional and transformer notebooks:

| Columns | Purpose |
| --- | --- |
| `text_clean` | The only dataset column sent to the model |
| `stressors`, `relationships`, `emotions`, `responses` | Reference annotations, separated by semicolons |
| `split` | Existing `train`, `val`, or `test` partition |
| `post_id` | A stable, unique post identifier |
| `orig_row_id` | Optional source-row metadata |
| `belong`, `burden`, `group` | TBe/PBu metadata for preparation and subgroup reporting |

The shared loader validates the table, removes empty text, and handles exact
text duplicates. It treats declared markers such as `not stated` as empty tag
sets. Unfinished annotations must be resolved before supervised evaluation.

The reference tags and TBe/PBu values are used for scoring or analysis; they
are not included in the classification prompt. Input CSV, TSV, and supported
Excel formats follow the shared [dataset guide](../data/README.md).

## 3. Choose development or final evaluation

Edit the settings near the top of `evaluate_ollama_llms.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DATA_FILE` | `data/processed/easy_dataset.csv` | Input path; `MINDMINER_DATA_FILE` can override it |
| `OUTPUT` | `results/ollama/` | Output location under the repository root |
| `MODELS` | The three model names above | Predeclared comparison models; Qwen names are rejected |
| `EVALUATION_SPLIT` | `"test"` | `"dev"` uses original train + val; `"test"` uses the separate test set |
| `MAX_POSTS` | `None` | All selected posts; a positive integer selects a seeded random subset |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service address |
| `SEED` | `42` | Sampling and generation seed |
| `NUM_CTX` | `8192` | Requested context window |
| `MAX_NEW_TOKENS` | `768` | Maximum generated output tokens |
| `MAX_POST_CHARACTERS` | `12000` | Reject longer posts without clipping their text |
| `FORMAT_RETRIES` | `2` | Two retries after the first format attempt; at most three attempts |
| `TIMEOUT_SECONDS` | `180` | Timeout configured on the Ollama client |

For a small setup check, use `EVALUATION_SPLIT="dev"` and `MAX_POSTS=5`.
Develop the prompt and settings using development data. Then freeze the
choices, set `EVALUATION_SPLIT="test"` and `MAX_POSTS=None`, and run the final
comparison. Do not repeatedly adjust the prompt in response to test scores.

The character limit is a guard, not an exact tokenizer count. The full prompt
and generated answer also consume context. The script does not silently
shorten a post to force a prediction.

## 4. Run the program

From the project root:

```bash
python scripts/evaluate_ollama_llms.py
```

The program checks that the configured models are installed and processes
them sequentially. For each post, it requests a JSON object containing the
four tag lists. It checks required fields, tag spelling, duplicate JSON keys,
repeated tags, and incomplete output. Malformed predictions receive bounded
format retries. Structural validity does not establish semantic correctness.

## 5. Understand the outputs

Each model's results use a folder such as:

```text
results/ollama/mistral_7b-instruct/test/<run_key>/
```

The run key identifies the input, model digest, settings, prompt, and script
version through hashes. Changed inputs or settings can therefore create a new
run folder instead of reusing an incompatible cache.

| File | Meaning |
| --- | --- |
| `run_settings.json` | Preparation audit, model digest, codebook identity, selected IDs, and generation settings |
| `prediction_cache.jsonl` | One saved prediction result per completed request; used for resuming |
| `completion.json` | Selected post count, valid predictions, and failures |
| `failed_predictions.json` | Failure details when any predictions are invalid |
| `overall_metrics.csv` | Overall multilabel scores for a fully completed model |
| `metrics_by_field.csv` | Scores for each of the four tag families |
| `per_label_metrics.csv` | Precision, recall, F1, support, and confusion counts per tag |
| `metrics_by_TBe_PBu_group.csv` | Tag-prediction scores within the TBe/PBu groups |
| `classification_report.txt` / `.csv` | Per-tag and averaged classification results |
| `predictions.csv` | Reference and predicted tags with IDs and analysis metadata |

The base output folder also contains a configuration-specific comparison CSV
and matching `.meta.json`. Check its completed and incomplete model lists
before treating a comparison as complete.

## 6. Interpret the metrics

| Metric | Beginner explanation |
| --- | --- |
| Precision | Of the tags predicted as present, how many match the reference? |
| Recall | Of the reference-present tags, how many did the model recover? |
| Macro F1 | Calculate F1 for every declared tag, then average equally |
| Micro F1 | Combine positive-decision counts across tags before calculating F1 |
| Exact-match accuracy | A post counts as correct only if every tag decision matches |
| Hamming loss | Fraction of post-tag decisions that are wrong; lower is better |

These JSON predictions are hard labels, not calibrated per-tag probabilities.
The script therefore does not calculate probability-based AP or ROC AUC.
The shared helper's zero-division and empty-set conventions are described in
[the results guide](../docs/RESULTS_GUIDE.md).

With `EVALUATION_SPLIT="dev"`, the script also summarizes existing predictions
across five development subsets. Files may use the shared helper's `cv_`
names, but **no model is retrained in those folds**. This is descriptive
subset reporting, not five-fold fine-tuning or nested cross-validation.

## 7. Resume and handle failures

Rerun the same script with unchanged settings to reuse valid cached
predictions. Previously unsuccessful entries are attempted again. An
interrupted or malformed cache line is skipped and can be predicted again.

A model with any failed posts does not receive benchmark metric reports for
that run. This prevents silently reporting only its successfully predicted
subset. Other completed models may still appear in the comparison table.
Transport or server exceptions can stop a run; after resolving the connection,
rerun it to reuse the cache.

## GPU use and reproducibility

Ollama manages model loading and GPU offloading on its server. The Python
client does not move an Ollama model with PyTorch `.to("cuda")`. The script
prints reported GPU memory and available `nvidia-smi` information; you can
also inspect loaded models with `ollama ps`. It requests unloading after each
model's completed pass.

Temperature is fixed at 0 and the generation seed is recorded. Keep model
digests, package versions, prompts, and run metadata with the results;
these settings alone do not guarantee identical outputs across every runtime.
Keep caches, quoted failure responses, and row-level predictions in the
ignored `results/` folder. Share reviewed aggregate tables through `reports/`.

References: [Ollama Python client](https://github.com/ollama/ollama-python),
[running-model information](https://docs.ollama.com/api/ps).
