# Notebook reading order

Each notebook starts with the research question, develops the basic ideas,
and places explanations beside the relevant code. The numbering is the
reading order; it does not force every notebook to use one input file.

| Notebook | Main input | Main output under `results/` |
| --- | --- | --- |
| [01_human_annotation_agreement.ipynb](01_human_annotation_agreement.ipynb) | An Excel workbook with independent annotator sheets | `agreement/human_agreement/` |
| [02_llm_annotation_agreement.ipynb](02_llm_annotation_agreement.ipynb) | An Excel workbook with two model annotation sheets | `agreement/llm_agreement/` |
| [03_exploratory_text_analysis.ipynb](03_exploratory_text_analysis.ipynb) | A text table in CSV, TSV, XLSX, or XLSM format | `eda/` |
| [04_traditional_text_classification.ipynb](04_traditional_text_classification.ipynb) | The processed modeling dataset | `traditional/` |
| [05_transformer_text_classification.ipynb](05_transformer_text_classification.ipynb) | The same processed modeling dataset | `transformers/` |

Read [START_HERE](../docs/START_HERE.md) for environment setup and the
[input guide](../data/README.md) for schemas. The two agreement notebooks
have separate sheet mappings and output folders. Public human-rater names
are aliases; actual workbook sheet names are configured in the settings.

Keep the complete `src/` folder with the notebooks. Opening a single
notebook without the shared modules is insufficient. Clear outputs before
committing an executed notebook that has displayed post content or IDs.
