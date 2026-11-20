<div align="center">

# Mobile game player spending segments

### CDS4003 · Data science for business applications

**A reproducible project on data quality, player-segment classification, and responsible campaign measurement.**

[Repository](https://github.com/MaryamBinHamdah/Mobile-Game-In-App-Purchases) · [Dataset source](https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025) · Streamlit app: *deployment link to be added after publishing*

</div>

---

## Project at a glance

This repository studies whether player profile and gameplay fields can classify a recorded in-app-purchase spending segment: **Minnow**, **Dolphin**, or **Whale**. The business scenario is NovaPlay Studios, a fictional free-to-play mobile-game publisher. A classifier may help a team form and test audience hypotheses. It cannot show that a promotion causes more purchases or produces profitable growth.

The work follows an ordered data-science workflow: inspect and clean the project dataset, describe segment patterns, create a small set of interpretable features, compare two model families, tune them with cross-validation, evaluate them on a held-out set, and demonstrate predictions in Streamlit.

> **Data limitation:** The Kaggle publisher describes the 3,024-row source dataset as synthetic. The 250,000-row project file is generated from that source by resampling and perturbing its records, with additional data-quality issues deliberately inserted. The extra rows are not new observations of real players. Model scores from this material are educational results only; they do not establish real-world performance.

## Business question and intended use

**Question:** Can non-purchase player profile and gameplay information classify a player's recorded spending segment?

The intended use is a classroom demonstration and a starting point for future, properly governed research. The project excludes direct spending amount and purchase-outcome fields from the prediction inputs. Even so, a valid operational forecast would require a clear prediction date, features measured before that date, and a genuinely later outcome window. The current source data do not provide that validated temporal design.

The 15% revenue increase mentioned in the project proposal is a proposed business target, not a measured result. Any real offer strategy would need a randomized holdout experiment and business measures such as incremental revenue after discount cost, contribution margin, purchase conversion, and retention.

## Project status

The repository includes the original source data, an expanded dataset for cleaning practice, data-cleaning and EDA scripts, model training and evaluation code, and a Streamlit demonstration. Run the commands below to generate the local outputs and model artifact. The Streamlit app is not considered live until a public deployment link is added and checked.

| Stage | Repository evidence |
|---|---|
| 1. Business problem | Project question and responsible-use framing in this README and the course report |
| 2. Data collection | `data/01_original_kaggle_data.csv`, source link, and `docs/02_dataset_search_evidence.pdf` |
| 3. Data understanding and cleaning | `src/clean_data.py`, `outputs/data_quality/cleaning_audit.json` after running |
| 4. EDA and feature engineering | `src/eda.py`, outputs under `outputs/eda/`, and notebook `notebooks/03_eda_feature_engineering.ipynb` |
| 5–6. Model selection, tuning, evaluation | `src/train_model.py`, `outputs/model_evaluation/`, and `models/model_metadata.json` after running |
| 7. Deployment demonstration | `app/main.py`; publish from this repository with Streamlit Community Cloud |

## Verified results from the current run

The cleaning script read 250,000 expanded rows and removed 2,500 exact duplicates, leaving 247,500 rows. Before imputation, the de-duplicated data contained 198,383 missing cells; the descriptive cleaning export contains no missing cells. This is a complete export, not proof that the imputed values were observed. The class counts after cleaning were 208,328 Minnows (84.17%), 33,670 Dolphins (13.60%), and 5,502 Whales (2.22%). These counts describe the generated project file.

The model script used the original 3,024-row file and a stratified 75/25 train/test split. Five-fold stratified cross-validation tuned two model families. The actual held-out results are:

| Model | CV macro-F1 | Test macro-F1 | Balanced accuracy | Test accuracy | Whale recall |
|---|---:|---:|---:|---:|---:|
| Random forest | 0.354 | 0.330 | 0.330 | 0.709 | 0.059 |
| Logistic regression | 0.266 | 0.257 | 0.331 | 0.384 | 0.294 |
| Majority-class baseline | — | 0.305 | 0.333 | 0.841 | 0.000 |

The random forest has the highest test macro-F1, so the app uses it for demonstration. It detects only 1 of 17 Whale cases in the test set (5.9% recall), and its balanced accuracy is slightly below the majority-class baseline. Logistic regression detects more Whales but performs poorly on the other classes. **Neither model is suitable for operational targeting.** The appropriate result is that this source and feature set do not provide a useful classifier under the current evaluation, not that the proposed business target has been achieved. These metrics are also based on a dataset the publisher describes as synthetic.

The numbers above are generated from `outputs/data_quality/cleaning_audit.json`, `outputs/eda/segment_distribution.csv`, and `outputs/model_evaluation/model_comparison.csv`. Re-run the documented commands before submission and confirm that the output files still show these values.

## Repository structure

```text
.
├── app/
│   └── main.py                         # Streamlit individual and batch scoring demo
├── data/
│   ├── 01_original_kaggle_data.csv     # Source dataset (3,024 rows)
│   ├── 02_expanded_dirty_data.zip      # Generated cleaning exercise data
│   └── processed/                      # Created locally by the cleaning script
├── docs/
│   ├── 01_data_dictionary.md
│   ├── 02_dataset_search_evidence.pdf
│   └── report_completion_checklist.md
├── images/                             # Project figures
├── models/                             # Created by model training; do not hand-edit
├── notebooks/                          # Ordered project notebooks
├── outputs/                            # Created locally by analysis scripts
├── src/
│   ├── clean_data.py                   # Stage 3: cleaning and quality audit
│   ├── eda.py                          # Stage 4: summary tables and figure
│   └── train_model.py                  # Stages 5–6: comparison and tuning
├── tests/
│   └── test_project.py                 # Basic workflow and artifact checks
├── requirements.txt
└── README.md
```

## Data and licensing

The source dataset is published on [Kaggle](https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025). The repository retains the source CSV and its documented dataset-search evidence. **Check the dataset's current licence and terms on its source page before reusing or redistributing it.** The course team's original code is covered by the repository `LICENSE`; third-party data and materials remain subject to their own terms.

The original file has 3,024 records and 13 fields. The expanded cleaning exercise has 250,000 rows and 23 columns, of which most rows are generated by resampling and perturbing source records. Therefore:

- The expanded file is useful for practicing cleaning at scale, but it does not satisfy a requirement for 200,000 independently collected real-world observations unless the instructor has explicitly approved this synthetic approach in writing.
- The model training script deliberately uses only the original 3,024-row source. This avoids presenting the generated expansion as independent evidence. The source itself is synthetic, so the model results still do not generalize to real users.
- Do not use synthetic records to make claims about real gamer behavior, commercial outcomes, or treatment of identifiable players.
- Do not commit credentials, personal data, or new data files without checking their size and permission.

See [`docs/01_data_dictionary.md`](docs/01_data_dictionary.md) for field descriptions and [`docs/02_dataset_search_evidence.pdf`](docs/02_dataset_search_evidence.pdf) for the team's dataset search documentation.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Run the stages in order from the repository root:

```bash
# Stage 3: clean the expanded dataset and save a quality audit
python src/clean_data.py

# Stage 4: generate descriptive summaries and a segment chart
python src/eda.py

# Stages 5–6: compare two models, tune with cross-validation, and evaluate
python src/train_model.py

# Stage 7: run the interactive app locally
streamlit run app/main.py

# Run the basic project checks
python -m unittest discover -s tests
```

Cleaning and EDA operate on the expanded project data. Model comparison operates on the original source file and saves its pipeline to `models/`. Preprocessing, including imputation and one-hot encoding, is part of the model pipeline so it is fitted on training data rather than the held-out test set. Outputs are created locally and are intentionally excluded from version control unless a specific, small evidence file is useful for review.

## Model evaluation

The training script compares multinomial logistic regression and random forest. It uses a stratified holdout set, stratified `GridSearchCV` on the training portion, macro-F1 for model selection, balanced accuracy, per-class precision and recall, and a confusion matrix. Macro-F1 gives each class equal weight; Whale recall is reported because Whales form a small part of the dataset. Accuracy is shown for context, not used alone.

After running the training script, review:

- `outputs/model_evaluation/model_comparison.csv` for tuned model scores;
- the model-specific classification reports and confusion matrices in the same folder; and
- `models/model_metadata.json` for the selected pipeline, parameters, and evaluation setup.

Do not copy a score into a report without checking the generated file and describing the synthetic-data limitation. A high score is not, by itself, evidence of business value.

## Streamlit deployment

1. Push the completed repository to GitHub and verify that the model files were generated by `python src/train_model.py`.
2. In Streamlit Community Cloud, create a new app connected to this repository and select `app/main.py` as the entry point.
3. Confirm the app starts, test a hypothetical profile, and test a CSV with the required feature columns.
4. Add the public app URL to the link near the top of this README and to the group report only after it is accessible.

The app accepts fictional single-profile inputs and batch CSV files. Do not upload identifiable or sensitive player records. The displayed model probabilities are not guarantees or estimates of customer value.

## Team and contribution evidence

The team and student details below reproduce the roles recorded in the supplied project draft. Members should verify their own contribution descriptions and add links to their real commits before submission. Do not claim work that was completed by another team member or by an AI tool.

| Member | Student ID | Reported project focus |
|---|---|---|
| Hessa Khalfan | H00535166 | Business problem framing, data generation, and data-quality logic |
| Nourah Abdulla | H00535834 | Data understanding, cleaning, and quality checks |
| Maryam Ali Bin Hamdah | H00535836 | Dataset search, selection, and data dictionary |

Use meaningful commits under each student's own GitHub account to evidence individual contributions. The individual written assessment and contribution statements must be prepared independently by each student, in their own words, as required by the course instructions.

## AI-tool disclosure

An AI assistant was used in this repository to support code development and debugging, including the cleaning, model-training, and Streamlit application code, and to improve technical repository documentation. Team members must review, test, understand, and accurately disclose any AI-assisted code they retain. AI-generated prose must not be submitted as the group's report or any student's individual written assessment. Those assessed texts must be written independently by the students, in accordance with the course instructions.

## References

- Chapman, P., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.* https://public.dhe.ibm.com/software/analytics/spss/documentation/modeler/14.2/es/CRISP-DM.pdf
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy online controlled experiments: A practical guide to A/B testing.* Cambridge University Press.
- Provost, F., & Fawcett, T. (2013). *Data science for business: What you need to know about data mining and data-analytic thinking.* O'Reilly Media.
- pratyushpuri. (n.d.). *Mobile game in-app purchases dataset 2025* [Data set]. Kaggle. https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025
