"""Shared data preparation, folds, and metrics for MindMiner.

Imported from src/ by the experiment notebooks.

Only text_clean is a model input. The annotation columns are reference labels;
belong (TBe), burden (PBu), and group are preserved as analysis metadata.
"""
from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, hamming_loss, jaccard_score,
    multilabel_confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)

SEED = 42
DATA_SCHEMA_VERSION = 2
FIELDS = ["stressors", "relationships", "emotions", "responses"]
# Fixed vocabulary from the supplied CSV: 10 + 7 + 8 + 5 = 30 binary labels.
# Never learn the vocabulary by looking at held-out test predictions.
TAGS = {
    "stressors": ["family conflict", "breakup or romance", "social isolation",
                  "financial", "study", "work", "health", "grief", "abuse", "other"],
    "relationships": ["partner", "family", "friend", "peer", "workplace", "community", "other"],
    "emotions": ["loneliness", "guilt", "shame", "sadness", "anxiety", "anger", "hopelessness", "other"],
    "responses": ["advice seeking", "emotional support seeking", "professional help", "withdrawal", "positive coping"],
}
LABELS = [f"{field}::{tag}" for field in FIELDS for tag in TAGS[field]]
DATA_COLUMNS = ["post_id", "orig_row_id", "split", "text_clean", "belong", "burden",
                "group", "stressors", "relationships", "emotions", "responses"]
META = ["post_id", "orig_row_id", "split", "belong", "burden", "group"]
REQUIRED_COLUMNS = [column for column in DATA_COLUMNS if column != "orig_row_id"]
EXCEL_ENGINES = {".xlsx": "openpyxl", ".xlsm": "openpyxl", ".xls": "xlrd", ".xlsb": "pyxlsb"}


def find_data_file(path=None):
    """Use an explicit path, or find the expected file in the current folder."""
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    names = ["easy_dataset.csv", "easy_dataset.tsv", "easy_dataset.xlsx", "easy_dataset.xls",
             "easy_dataset.xlsm", "easy_dataset.xlsb",
             "Easy version (4).csv", "Easy version (4).xlsx"]
    matches = [Path(name) for name in names if Path(name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("Set DATA_FILE to your CSV/Excel path, or keep exactly one expected data file beside the scripts.")
    return matches[0]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def package_versions():
    names = ["numpy", "pandas", "scikit-learn", "iterative-stratification", "torch", "transformers", "ollama"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def clean_text(value):
    # Light cleaning only: keep negation, punctuation, pronouns, and word order.
    value = unicodedata.normalize("NFKC", html.unescape(str(value)))
    return re.sub(r"\s+", " ", value).strip()


def split_tags(value):
    # In this annotated export a blank label cell means no tag in that field.
    # If blanks instead mean unannotated, complete annotation before modeling.
    if pd.isna(value) or not str(value).strip():
        return []
    return sorted(set(re.sub(r"\s+", " ", part.strip().lower())
                      for part in str(value).split(";") if part.strip()))


def _review_flag(value):
    """Optional legacy flags: keep explicit false values; exclude true/unknown status."""
    if pd.isna(value):
        return True
    return str(value).strip().casefold() not in {"false", "0", "0.0", "no", "n"}


def _has_validation_issue(value):
    if pd.isna(value):
        return False
    return str(value).strip().casefold() not in {"", "none", "nan", "null", "[]", "{}"}


def _text_keys(frame):
    return set(frame["text_clean"].str.casefold())


def load_data(path):
    """Read the 11-column export by column name, preserving IDs and TBe/PBu metadata.

    Review columns are optional. When available, filter flagged rows before dropping
    those columns. Without them, use the supplied export and record that no flag-based
    filtering could be applied. orig_row_id is optional for older exports only.
    """
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix in EXCEL_ENGINES:
        raw = pd.read_excel(path, dtype=object, engine=EXCEL_ENGINES[suffix])
    elif suffix in {".csv", ".tsv"}:
        raw = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=object, encoding="utf-8-sig")
    else:
        raise ValueError("Use .csv, .tsv, .xlsx, .xlsm, .xls or .xlsb.")
    original_n = len(raw)
    raw = raw.dropna(how="all").copy()
    empty_rows_removed = original_n - len(raw)
    raw.columns = [unicodedata.normalize("NFKC", str(column)).strip().lstrip("\ufeff").strip().casefold()
                   for column in raw.columns]
    if raw.columns.duplicated().any():
        raise ValueError("Column names are duplicated after removing whitespace/case differences.")
    source_columns = raw.columns.tolist()
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    review_columns = [column for column in ["label_needs_review", "label_validation_issues"] if column in raw]
    excluded = pd.Series(False, index=raw.index)
    if "label_needs_review" in raw:
        excluded |= raw["label_needs_review"].map(_review_flag)
    if "label_validation_issues" in raw:
        excluded |= raw["label_validation_issues"].map(_has_validation_issue)
    review_rows_removed = int(excluded.sum())
    raw = raw.loc[~excluded].copy()
    orig_row_id_available = "orig_row_id" in raw
    if not orig_row_id_available:
        raw["orig_row_id"] = pd.NA  # Missing optional metadata is not replaced by an invented ID.
    # Exactly the new 11-column layout; all other annotation columns are discarded here.
    df = raw[DATA_COLUMNS].copy()
    before_text_cleaning = len(df)
    df = df[df["text_clean"].notna()].copy()
    df["text_clean"] = df["text_clean"].map(clean_text)
    df = df[df["text_clean"].ne("")].copy()
    df["post_id"] = df["post_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    df["orig_row_id"] = df["orig_row_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if df["post_id"].isna().any() or df["post_id"].eq("").any():
        raise ValueError("Every nonempty post needs a post_id.")
    if df["post_id"].duplicated().any():
        raise ValueError("post_id must identify one row. Resolve repeated IDs first.")
    df["split"] = df["split"].astype("string").str.strip().str.casefold().replace({"validation": "val", "valid": "val"})
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("split must contain train, val, or test.")
    for col in ["belong", "burden"]:
        values = pd.to_numeric(df[col], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"{col} must contain 0/1, without missing values.")
        df[col] = values.astype(int)
    expected_group = np.select(
        [(df.belong == 1) & (df.burden == 1), df.belong == 1, df.burden == 1],
        ["Both", "TBe only", "PBu only"], default="Neither",
    )
    observed_group = df["group"].astype("string").str.strip().str.casefold().map({
        "tbe only": "TBe only", "pbu only": "PBu only", "both": "Both", "neither": "Neither"})
    if observed_group.isna().any() or not (observed_group.to_numpy() == expected_group).all():
        raise ValueError("group disagrees with belong/burden. Resolve it before modeling.")
    df["group"] = expected_group
    for field in FIELDS:
        parsed = df[field].map(split_tags)
        unknown = sorted(set(tag for tags in parsed for tag in tags) - set(TAGS[field]))
        if unknown:
            raise ValueError(f"Unknown {field} labels: {unknown}. Update the declared vocabulary deliberately.")
        df[field] = parsed.map(lambda tags: "; ".join(tags))
    df["_text_key"] = df["text_clean"].str.casefold().map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    duplicate_rows = df[df["_text_key"].duplicated(keep=False)]
    for _, rows in duplicate_rows.groupby("_text_key"):
        if rows[FIELDS + ["belong", "burden"]].drop_duplicates().shape[0] > 1:
            raise ValueError("Identical texts have conflicting annotations. Resolve those annotations first.")
    # Preserve the original test copy; remove its development-set copies.
    df["_priority"] = df["split"].map({"train": 0, "val": 1, "test": 2})
    before = len(df)
    df = df.sort_values("_priority", kind="stable").drop_duplicates("_text_key", keep="last").sort_index()
    audit = {"data_schema_version": DATA_SCHEMA_VERSION,
             "source_rows": original_n, "empty_rows_removed": empty_rows_removed,
             "empty_text_rows_removed": before_text_cleaning - before,
             "review_columns_present": review_columns, "review_filter_applied": bool(review_columns),
             "review_rows_removed": review_rows_removed,
             "input_columns": source_columns, "retained_columns": DATA_COLUMNS,
             "orig_row_id_available": orig_row_id_available,
             "duplicates_removed": before - len(df), "retained_posts": len(df),
             "source_sha256": file_hash(path), "split_counts": df["split"].value_counts().to_dict(),
             "label_count": len(LABELS), "versions": package_versions()}
    df = df.drop(columns=["_priority", "_text_key"]).reset_index(drop=True)
    dev = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"].eq("test")].reset_index(drop=True)
    if dev.empty or test.empty:
        raise ValueError("Both a development partition (train/val) and a test partition are required.")
    assert _text_keys(dev).isdisjoint(_text_keys(test))
    print(f"Posts: {len(df)} | development (train + val): {len(dev)} | untouched test: {len(test)}")
    print(f"Targets: {len(LABELS)} | duplicate copies removed: {before - len(df)}")
    print(f"Review/validation filtering: {review_rows_removed} rows removed" if review_columns
          else "Review/validation columns absent: using the supplied 11-column export.")
    return df, dev, test, audit


def encode_labels(df):
    """Each label gets its own 0/1 column; a post may have multiple labels."""
    columns = []
    for field in FIELDS:
        rows = df[field].map(split_tags)
        for tag in TAGS[field]:
            columns.append(rows.map(lambda tags: int(tag in tags)).to_numpy())
    return np.column_stack(columns).astype(np.int64)


def make_folds(df, y, n_splits=5, seed=SEED):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    # Balance all 30 labels plus TBe/PBu groups. Metadata is used for splitting,
    # never as a predictive feature. Text duplicates were removed above.
    groups = pd.get_dummies(df["group"]).to_numpy(dtype=int)
    stratification_labels = np.column_stack([y, groups])
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(df)), stratification_labels))


def save_fold_plan(df, y, folds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = df[META].copy()
    plan["cv_fold"] = -1
    counts = []
    for number, (train_idx, val_idx) in enumerate(folds, start=1):
        assert _text_keys(df.iloc[train_idx]).isdisjoint(_text_keys(df.iloc[val_idx]))
        plan.loc[val_idx, "cv_fold"] = number
        for j, label in enumerate(LABELS):
            counts.append({"fold": number, "label": label,
                           "train_positives": int(y[train_idx, j].sum()),
                           "validation_positives": int(y[val_idx, j].sum())})
    assert (plan.cv_fold > 0).all()
    plan.to_csv(out_dir / "cv_fold_assignments.csv", index=False)
    pd.DataFrame(counts).to_csv(out_dir / "cv_label_counts.csv", index=False)


def metrics(y_true, y_pred, scores=None):
    result = {"n_posts": len(y_true), "exact_match_accuracy": accuracy_score(y_true, y_pred),
              "hamming_loss": hamming_loss(y_true, y_pred),
              "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=1)}
    for average in ["micro", "macro", "weighted", "samples"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        result.update({f"precision_{average}": p, f"recall_{average}": r, f"f1_{average}": f})
    if scores is not None:
        has_positive = y_true.sum(axis=0) > 0
        has_both = has_positive & (y_true.sum(axis=0) < len(y_true))
        result["average_precision_macro"] = float(np.mean([
            average_precision_score(y_true[:, j], scores[:, j]) for j in np.flatnonzero(has_positive)
        ])) if has_positive.any() else np.nan
        result["roc_auc_macro"] = float(np.mean([
            roc_auc_score(y_true[:, j], scores[:, j]) for j in np.flatnonzero(has_both)
        ])) if has_both.any() else np.nan
        result["ap_supported_labels"] = int(has_positive.sum())
        result["auc_supported_labels"] = int(has_both.sum())
    return result


def save_classification_reports(y_true, y_pred, out_dir, print_report=True):
    """Save sklearn reports for all 30 labels and each of the four fields."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in ["all_labels"] + FIELDS:
        if field == "all_labels":
            cols = list(range(len(LABELS)))
            names = LABELS
            filename = "classification_report"
        else:
            cols = [j for j, label in enumerate(LABELS) if label.startswith(field + "::")]
            names = TAGS[field]
            filename = f"classification_report_{field}"
        # Multilabel rows are tags, followed by micro/macro/weighted/samples
        # averages. Support counts positive reference labels, not all posts.
        text = classification_report(
            y_true[:, cols], y_pred[:, cols], target_names=names,
            digits=4, zero_division=0,
        )
        report_dict = classification_report(
            y_true[:, cols], y_pred[:, cols], target_names=names,
            output_dict=True, zero_division=0,
        )
        (out_dir / f"{filename}.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(report_dict).T.rename_axis("label").to_csv(out_dir / f"{filename}.csv")
        if print_report and field == "all_labels":
            print(f"\nClassification report — {out_dir}\n{text}")


def write_evaluation(df, y_true, y_pred, scores, out_dir):
    """Save overall, per-family, per-label, group, and row-level results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics(y_true, y_pred, scores)
    save_classification_reports(y_true, y_pred, out_dir)
    pd.DataFrame([summary]).to_csv(out_dir / "overall_metrics.csv", index=False)
    family_rows = []
    for field in FIELDS:
        cols = [j for j, label in enumerate(LABELS) if label.startswith(field + "::")]
        family_rows.append({"field": field, **metrics(y_true[:, cols], y_pred[:, cols], None if scores is None else scores[:, cols])})
    pd.DataFrame(family_rows).to_csv(out_dir / "metrics_by_field.csv", index=False)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    counts = multilabel_confusion_matrix(y_true, y_pred).reshape(-1, 4)
    per_label = pd.DataFrame({"label": LABELS, "precision": p, "recall": r, "f1": f, "support": support,
                              "TN": counts[:, 0], "FP": counts[:, 1], "FN": counts[:, 2], "TP": counts[:, 3]})
    per_label.to_csv(out_dir / "per_label_metrics.csv", index=False)
    group_rows = []
    for group in ["TBe only", "PBu only", "Both", "Neither"]:
        mask = df["group"].eq(group).to_numpy()
        if mask.any():
            group_rows.append({"group": group, **metrics(y_true[mask], y_pred[mask])})
    pd.DataFrame(group_rows).to_csv(out_dir / "metrics_by_TBe_PBu_group.csv", index=False)
    # Keep TBe/PBu metadata in predictions; do not export post text by default.
    predictions = df[META].reset_index(drop=True).copy()
    for field in FIELDS:
        cols = [j for j, label in enumerate(LABELS) if label.startswith(field + "::")]
        tags = TAGS[field]
        predictions["true_" + field] = ["; ".join(tag for tag, bit in zip(tags, row[cols]) if bit) for row in y_true]
        predictions["pred_" + field] = ["; ".join(tag for tag, bit in zip(tags, row[cols]) if bit) for row in y_pred]
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    if scores is not None:
        score_frame = pd.DataFrame(scores, columns=LABELS)
        score_frame.insert(0, "post_id", df["post_id"].to_numpy())
        score_frame.to_csv(out_dir / "label_scores.csv", index=False)
    return summary


def summarize_cv(rows, out_dir):
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "cv_metrics_each_fold.csv", index=False)
    names = [name for name in frame.select_dtypes(include=np.number).columns if name not in ["fold", "n_posts"]]
    summary = frame.groupby("model")[names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("f1_macro_mean", ascending=False)
    summary.to_csv(Path(out_dir) / "cv_comparison_mean_std.csv")
    return summary
