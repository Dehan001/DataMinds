# Read the experiment outputs

These files are produced when you execute the notebook. Their presence in
this guide does not mean an experiment has already been run. Outputs use
`results/transformers/` as their base folder.

## Start with three questions

1. **Which model was chosen before test inspection?** Read
   `selected_by_cv.json` and `cv_comparison_mean_std.csv`.
2. **How well did the fixed experiments perform on test?** Read
   `test_comparison.csv` and each model's `test/` reports.
3. **Where are the errors?** Read per-field, per-label, and subgroup metrics.

## Shared experiment files

| File | What it tells you |
| --- | --- |
| `run_settings.json` | Dataset hash, preparation audit, available package versions, labels, settings, and device |
| `cv_fold_assignments.csv` | Each development post's fold number and metadata; keep local |
| `cv_label_counts.csv` | Positive examples per label in each fold's training and validation partitions |
| `cv_metrics_each_fold.csv` | Each model's individual development-fold scores |
| `cv_comparison_mean_std.csv` | Fold mean and sample standard deviation for each model |
| `selected_by_cv.json` | The model selected using mean development macro F1 |
| `test_comparison.csv` | Final test metrics for the three predeclared models |
| `cv_macro_f1.png` | Development macro F1 bars with fold standard-deviation error bars |

## Files inside each model's folder

`<model>` means `BERT`, `RoBERTa`, or `DistilBERT`.

| Path under `<model>/` | What it tells you |
| --- | --- |
| `token_lengths.json` | Development-token length summary and number of truncated posts |
| `fold_N_model.json` | Model ID, resolved model commit if available, and seed for that fold |
| `final_model.json` | Model ID, resolved model commit if available, and seed for the final development fit |
| `fold_N/classification_report.*` | Per-label and average metrics for that held-out fold |
| `development_oof/` | Pooled out-of-fold evaluation, with one held-out prediction per development post |
| `test/` | Evaluation after the final fit on all development rows |
| `saved_model/` | Model and tokenizer files, only if saving is enabled |

## Files in `development_oof/` and `test/`

| File | Meaning |
| --- | --- |
| `overall_metrics.csv` | Overall multilabel metrics |
| `metrics_by_field.csv` | Separate scores for stressors, relationships, emotions, and responses |
| `per_label_metrics.csv` | Precision, recall, F1, support, and TN/FP/FN/TP for each tag |
| `metrics_by_TBe_PBu_group.csv` | Tag-prediction performance within each TBe/PBu subgroup |
| `classification_report.txt` / `.csv` | All 30 tags and supported multilabel averaging rows |
| `classification_report_<field>.txt` / `.csv` | A report for one of the four tag families |
| `predictions.csv` | Reference and predicted tag sets with IDs and TBe/PBu metadata; keep local |
| `label_scores.csv` | One continuous score per tag and post ID; keep local |

Subgroup scores here measure the four tag families **within groups**.
They do not measure the accuracy of predicting TBe/PBu itself.

## The metrics in plain language

| Metric | What it means | Better direction |
| --- | --- | --- |
| Precision | How many predicted positives were correct? | Higher |
| Recall | How many reference positives were recovered? | Higher |
| F1 | Balance of precision and recall | Higher |
| Macro average | Calculate a metric for each tag, then average equally | Higher for precision/recall/F1 |
| Micro average | Pool tag-level TP/FP/FN counts before calculating | Higher for precision/recall/F1 |
| Weighted average | Average tag scores using positive support as weights | Higher for precision/recall/F1 |
| Samples average | Calculate within each post, then average over posts | Higher for precision/recall/F1 |
| Exact-match accuracy | Fraction of posts with every tag decision correct | Higher |
| Hamming loss | Fraction of post–tag decisions that are wrong | Lower |
| Sample Jaccard | Overlap of predicted and reference tag sets divided by their union, averaged over posts | Higher |
| Macro average precision (AP) | Summarize precision across score thresholds for each supported tag, then average | Higher |
| Macro ROC AUC | Measure positive-versus-negative ranking for each supported tag, then average | Higher |
| Support | Number of reference-positive occurrences of a tag | A count, not performance |

A high exact-match score is demanding: missing just one tag makes that post
an incorrect exact match. A low Hamming loss can coexist with weak positive
detection when most of the 30 decisions are negative. Read several metrics
together, especially per-label recall and macro F1.

## Conventions that affect interpretation

- Precision/recall/F1 use `zero_division=0`. The fixed vocabulary includes
  unsupported tags in macro F1; undefined scores for those tags become 0.
- Jaccard uses `zero_division=1`, so a post with no reference tags and no
  predicted tags receives a Jaccard score of 1.
- Average precision includes only tags with at least one reference positive.
  ROC AUC includes only tags with both positive and negative examples.
  `ap_supported_labels` and `auc_supported_labels` record those counts.
- Pooled out-of-fold F1 is calculated from pooled predictions. It need not
  equal the mean of five fold-level F1 scores.
- Error bars in `cv_macro_f1.png` represent the sample standard deviation
  across five folds. They are not confidence intervals.

## Describe your actual run

A methods statement should give the input dataset version/hash, the four
tag families and declared label count, fixed test partition, preprocessing,
five-fold development procedure, model identifiers, exact training settings,
decision threshold, selection metric, and metric conventions.

Replace descriptions with actual logged values when settings change. Do not
report nested cross-validation unless you implement inner-fold tuning. Keep
test results separate from the development scores used to choose a model.

Source definitions: [scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).
