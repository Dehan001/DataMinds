# Dataset instructions

Dataset bytes are not included in this package. Keep your approved local
files in the folders below; their contents are ignored by Git.

| Folder | What it means |
| --- | --- |
| `raw/` | Original source exports; preserve as provenance |
| `annotations/` | Intermediate annotation and label exports |
| `processed/` | The chosen modeling dataset, normally `easy_dataset.csv` |
| `splits/` | Existing train/validation/test files kept for provenance |

## Expected columns

The shared helper selects columns by name, so their order does not matter.
It normalizes whitespace/case in column names and rejects duplicate names.

| Column | Expected content | Role |
| --- | --- | --- |
| `post_id` | A nonempty unique ID for each row | Identity; not a feature |
| `orig_row_id` | Original row ID if available | Optional provenance; not a feature |
| `split` | `train`, `val`, or `test` | Existing partition |
| `text_clean` | Nonempty post text | The only model input |
| `belong` | 0 or 1 | TBe metadata |
| `burden` | 0 or 1 | PBu metadata |
| `group` | `TBe only`, `PBu only`, `Both`, or `Neither` | Must agree with the two binary columns |
| `stressors` | Zero or more declared tags, separated by semicolons | Target annotations |
| `relationships` | Same format | Target annotations |
| `emotions` | Same format | Target annotations |
| `responses` | Same format | Target annotations |

`orig_row_id` is optional for older exports and is left missing when absent.
`validation` and `valid` in the split column are normalized to `val`.
The group meanings are:

| belong | burden | group |
| --- | --- | --- |
| 1 | 0 | TBe only |
| 0 | 1 | PBu only |
| 1 | 1 | Both |
| 0 | 0 | Neither |

## Exact vocabulary declared by the helper

| Field | Count | Tags |
| --- | --- | --- |
| stressors | 10 | family conflict; breakup or romance; social isolation; financial; study; work; health; grief; abuse; other |
| relationships | 7 | partner; family; friend; peer; workplace; community; other |
| emotions | 8 | loneliness; guilt; shame; sadness; anxiety; anger; hopelessness; other |
| responses | 6 | advice seeking; emotional support seeking; professional help; withdrawal; positive coping |

These are 30 separate binary labels. For example, `stressors::other` and
`emotions::other` are different labels even though their final word matches.
Confirm this vocabulary against the dataset and current codebook before a
real run. Unknown tags cause an error rather than being silently discarded.

A blank annotation cell means **no tag in that family** in this loader.
If a blank means “not annotated” in your file, resolve that distinction
before using it for model training.

## What preparation does

The helper removes empty rows and empty text, normalizes Unicode and HTML
escapes, and compresses repeated whitespace. It retains punctuation,
negation, pronouns, and word order. It parses semicolon-separated tags and
validates IDs, split names, metadata, and label spelling.

Optional legacy review columns are used to exclude flagged records before
those columns are dropped. If the columns are absent, that filter is not
applied and the audit records this. Unrecognized or missing review-flag
values are treated conservatively as flagged by the supplied helper.

Duplicate detection compares the normalized, case-folded full text. Conflicting
annotations for identical texts cause an error. For consistent duplicate
texts, one copy is retained with the priority **test, then val, then train**.
Repeated `post_id` values are rejected before this text-deduplication step.
The loader does not detect every paraphrase or near-duplicate.

After preparation, original `train` plus `val` form development; `test`
remains separate. Actual row counts come from your file at run time. The
notebook never reads the three files in `data/splits/` automatically.

## Supported formats

CSV and TSV are supported, as are `.xlsx`, `.xlsm`, `.xls`, and `.xlsb`.
`openpyxl` is included for `.xlsx`/`.xlsm`. Install `xlrd` for `.xls` or
`pyxlsb` for `.xlsb` only if you use those formats. Preserve the real
extension and set `DATA_FILE` to the file you actually have.
