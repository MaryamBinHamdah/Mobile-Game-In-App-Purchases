# CDS4003 report completion checklist

Use this checklist to plan and verify the group's report and each student's individual assessment. It is a work aid, not report prose. Write the assessed narrative independently, in your own words, and check the course instructions before submission.

## Before finalizing the project

- [ ] Confirm with the instructor, in writing, whether the Kaggle source and the expanded synthetic dataset meet Milestone 2 requirements. The source has 3,024 rows; its publisher describes it as synthetic; the expanded 250,000-row file contains generated records.
- [ ] Confirm whether synthetic data are permitted for this project and obtain written approval if required. The supplied rubric specifies instructor approval for synthetic data.
- [ ] Verify the dataset licence and permitted academic use on the Kaggle source page. Do not rely only on the draft's CC0 statement.
- [ ] Confirm the prediction time, feature observation window, and target outcome window. Remove fields unavailable by the prediction time and any field used to define the target.
- [ ] Run all scripts from a clean checkout and retain the actual output files as evidence.
- [ ] Verify the Streamlit deployment URL, GitHub visibility, and the final commit history.
- [ ] Ask each team member to verify their own name, ID, role, and contribution evidence.
- [ ] Include the required AI-tool disclosure and follow the course's academic-integrity rules.

## Group report: evidence to collect for each stage

### Stage 1: business problem formulation

- [ ] Identify the business context and intended users of the analysis.
- [ ] State a specific data-science question and operational decision.
- [ ] Distinguish the proposed 15% revenue target from a measured project result.
- [ ] Define how business impact would be measured in a future experiment, including incremental revenue, offer cost, contribution margin, and retention.
- [ ] Explain what a spending-segment classifier can and cannot establish.

### Stage 2: data collection and selection

- [ ] Cite the source, date accessed, file name, dimensions, target, and licence evidence.
- [ ] Show the search terms and candidate datasets that were accepted or rejected.
- [ ] State clearly which records are original source records and which are generated.
- [ ] Provide the instructor's approval evidence if synthetic data were used.
- [ ] Explain why 250,000 generated rows are not 250,000 independently collected observations.

### Stage 3: data understanding and cleaning

- [ ] Run `python src/clean_data.py` and use `outputs/data_quality/cleaning_audit.json` for the before/after counts.
- [ ] Report observed duplicate counts, missingness, inconsistent formats, range violations, and date issues from the actual audit.
- [ ] Explain why each invalid value was corrected, treated as missing, or left unchanged.
- [ ] Distinguish descriptive-data imputation from model preprocessing. The model pipeline learns preprocessing from training data only.
- [ ] Validate the final CSV for duplicates, missing values, valid target labels, and applicable ranges.

### Stage 4: EDA and feature engineering

- [ ] Run `python src/eda.py` and link each chart or table to a specific question.
- [ ] Report the segment shares and consider the imbalance when reading later model results.
- [ ] Compare within-segment percentages, not only raw counts, for categorical variables.
- [ ] Report effect size and practical size of differences; large row counts can make small differences statistically significant.
- [ ] Explain the definition and intended meaning of every engineered variable.
- [ ] Describe associations as patterns, not causes. Generated-data results do not describe actual player behavior.

### Stages 5–6: modeling, tuning, and evaluation

- [ ] Run `python src/train_model.py` and cite `outputs/model_evaluation/model_comparison.csv`.
- [ ] Explain why both candidate models suit the task and identify the baseline.
- [ ] Report the held-out test set size, stratified split, cross-validation method, tuning metric, and selected parameters.
- [ ] Interpret macro-F1, balanced accuracy, per-class precision and recall, Whale recall, and the confusion matrix.
- [ ] Avoid selecting on accuracy alone. Check the majority-class baseline and whether the smaller classes are detected.
- [ ] Verify no purchase-derived feature directly reveals the target. Explain the remaining synthetic-data and timing limitations.
- [ ] Do not describe a score on the synthetic source as performance on real players.

### Stage 7: deployment

- [ ] Confirm the app loads from `app/main.py` and the saved model artifact is available in the repository used for deployment.
- [ ] Test a hypothetical single profile and a small CSV batch. Record the accepted input columns and output behavior.
- [ ] Add the live Streamlit URL only after opening and testing it in a fresh browser session.
- [ ] Describe the warning, privacy, and responsible-use controls in the app.
- [ ] State that model classification is not an offer-uplift model and does not prove revenue growth.

## Individual written assessment: each student completes independently

The rubric calls for a 2,000–3,000-word individual report covering the seven stages, a 500–800-word critical reflection, and an individual contribution statement of at least 300 words. Each student must write these independently and use their own evidence and reasoning.

- [ ] Explain and justify personal understanding of the business problem and dataset choice.
- [ ] Explain data cleaning, EDA, and feature-engineering decisions using verified project evidence.
- [ ] Explain model choice, tuning, evaluation, and the meaning of the actual results.
- [ ] Explain deployment and realistic business value, with limitations and recommendations.
- [ ] In the reflection, discuss real challenges, considered alternatives, limitations, and improvements.
- [ ] In the contribution statement, identify exact work completed and evidence it with the student's own GitHub commits, notebook changes, or files.
- [ ] Do not copy another student's text or use AI-generated prose in the assessed submission. Follow the instructor's directions on AI-assisted code and disclosure.

## Final consistency check

- [ ] Every count, chart, score, and claim matches a generated file or a cited source.
- [ ] The README, app, notebooks, and report use the same dataset description and prediction objective.
- [ ] No synthetic rows are described as real people or independent observations.
- [ ] No unsupported claim says the project achieved a revenue increase or caused a business outcome.
- [ ] The group report's first page contains the required GitHub and working Streamlit links.
- [ ] Each student checks word counts, references, declaration, and submission requirements before uploading.
