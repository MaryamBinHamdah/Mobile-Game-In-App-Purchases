<div align="center">

# 🎮 Mobile Game Player Spending Segmentation

### Player Intelligence · CDS4003 Data Science Project


<img width="2688" height="1152" alt="Player segmentation project hero" src="https://github.com/user-attachments/assets/3936ddc0-1a28-41b4-8559-cef9934d80bd" />

### Turning player-behaviour data into smarter, more timely engagement

![Predictive Analytics](https://img.shields.io/badge/ANALYTICS-PREDICTIVE%20ANALYTICS-6D28D9?style=for-the-badge)
![Classification](https://img.shields.io/badge/ML%20TASK-MULTI--CLASS%20CLASSIFICATION-0891B2?style=for-the-badge)
![Project Status](https://img.shields.io/badge/STATUS-IN%20PROGRESS-16A34A?style=for-the-badge)

**A data-science project that classifies mobile-game players as Minnow, Dolphin, or Whale to help NovaPlay plan and test more relevant promotions.**

[Project summary](#about-the-project) · [Data](#data-at-a-glance) · [Workflow](#project-workflow) · [Results](#model-results) · [Deployment](#deployment) 

</div>

---

## About the project

NovaPlay Studios is the business scenario for this project. Its marketing and live-operations teams want to understand player-spending segments and use them to define more relevant promotional tests.

> **Business question:** Can profile and gameplay information help classify a player as Minnow, Dolphin, or Whale so NovaPlay can test player-specific promotional strategies?

The business objective is to evaluate whether targeted campaigns could support a **15% increase in in-app-purchase (IAP) revenue within three months of deployment**, while protecting contribution margin and player retention. The 15% is a project target, **not a result achieved by the model**. The classifier predicts a spending segment; it does not predict an exact spending amount or prove that a promotion causes revenue growth.

## Data at a glance

The project uses a Kaggle dataset that its listing describes as synthetic. The team expanded it with additional synthetic records and documented the data preparation steps. The expanded data include intentionally introduced quality issues for cleaning practice. The original Kaggle records and generated rows are kept distinct in the project documentation.

<div align="center">
<table align="center">
<thead><tr><th align="center">Dataset / resource</th><th align="center">Size or purpose</th></tr></thead>
<tbody>
<tr><td align="center"><a href="data/01_original_kaggle_data.csv">Original Kaggle CSV</a> · <a href="https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025">source page</a></td><td align="center">3,024 rows × 13 columns</td></tr>
<tr><td align="center"><a href="data/02_expanded_dirty_data.zip">Expanded, uncleaned data</a></td><td align="center">250,000 rows × 23 columns</td></tr>
<tr><td align="center"><a href="data/03_clean_data.zip">Cleaned data</a></td><td align="center">247,500 rows × 23 columns</td></tr>
<tr><td align="center"><a href="data/04_mobile_eda_features.zip">EDA feature dataset</a></td><td align="center">Prepared features for analysis and modeling</td></tr>
<tr><td align="center"><a href="docs/01_data_dictionary.md">Data dictionary</a></td><td align="center">Field definitions and data notes</td></tr>
<tr><td align="center"><a href="docs/02_dataset_search_evidence.pdf">Dataset search evidence</a></td><td align="center">Dataset search and selection documentation</td></tr>
</tbody></table>
</div>

The spending target is imbalanced: **84.17% Minnow, 13.60% Dolphin, and 2.22% Whale**. For that reason, accuracy is considered together with balanced accuracy, Macro-F1, and class-specific results.

## Project workflow

1. **Business problem:** define the NovaPlay scenario, measurable revenue objective, target segments, and business constraints.
2. **Data collection and preparation:** document the Kaggle source, expand the data for the project, and record the dataset dictionary and search evidence. See [`01_data_generation.ipynb`](notebooks/01_data_generation.ipynb).
3. **Data understanding and cleaning:** inspect and address missing values, duplicates, inconsistent formats, invalid values, and outliers. See [`02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb).
4. **EDA and feature engineering:** explore segment and engagement patterns and create interpretable 30-day behavioural features. See [`03_eda_feature_engineering.ipynb`](notebooks/03_eda_feature_engineering.ipynb).
5. **Model development:** compare Decision Tree, XGBoost, Random Forest, a majority-class Dummy baseline, and a supplementary TF-IDF + Logistic Regression experiment. See the two modeling notebooks in [`notebooks/`](notebooks/).
6. **Tuning and evaluation:** tune models using cross-validation, evaluate them on the held-out test split, and review class-specific trade-offs.
7. **Deployment:** publish the selected Gini Decision Tree through Streamlit Community Cloud. The app accepts player inputs, shows a segment prediction, and demonstrates how that output can inform a controlled offer test.

## Model results

The following are the project’s **held-out test results**. They describe performance on the project dataset, not guaranteed performance on future or live players.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| **Decision Tree — Gini (selected)** | **94.62%** | **86.23%** | **86.65%** |
| Decision Tree — Entropy, GridSearchCV-selected | 94.32% | 85.90% | 86.05% |
| Random Forest — tuned | 91.78% | 92.22% | 78.64% |
| XGBoost — tuned | 88.15% | 91.88% | 76.51% |
| Dummy — majority-class baseline | 84.17% | 33.33% | 30.47% |
| TF-IDF + Logistic Regression | 43.02% | 46.30% | 30.18% |

The Gini Decision Tree is used in the deployed app. Its three-fold cross-validation Macro-F1 was **0.8450**, close to the GridSearchCV-selected Entropy tree (**0.8492**). Gini achieved the higher held-out Macro-F1 and **81.05% Whale precision**. The tuned Random Forest and XGBoost found more Whales (recall of 94.45% and 95.18%), but with lower Whale precision (43.90% and 48.97%). This is a business trade-off: higher Whale recall can identify more high-spending players, while lower precision can result in more non-Whales being included in a Whale-targeted offer group. The Gini tree was selected for the app based on its overall and class-specific test performance and the project’s margin-conscious use case.

The majority-class baseline shows why accuracy alone is insufficient: it reaches 84.17% accuracy by predicting the most common segment, but its balanced accuracy is 33.33% and it identifies no Dolphin or Whale players.

## App overview

The app uses six inputs: age, country, device, favourite game genre, session count, and average session length. It displays the predicted segment, the model score for that input, scores across all segments, a segment-based offer idea, and a global feature-importance summary. A player’s model score is **not** the model’s overall accuracy and is not a guarantee of future spending. Feature importance describes the fitted model overall; it is not a causal explanation for an individual player.

![Player profile inputs and predicted spending segment in the deployed app](https://private-us-east-1.manuscdn.com/sessionFile/QAOnFAiMQMuaV9NPtCpc2f/sandbox/gQDNCpW9BGiido1gGWIpqn-images_1795458445009_na1fn_L2hvbWUvdWJ1bnR1L3NoYXJlL25vdmFwbGF5X3N0cmVhbWxpdC9pbWFnZXMvc3RyZWFtbGl0X3BsYXllcl9wcmVkaWN0aW9u.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUUFPbkZBaU1RTXVhVjlOUHRDcGMyZi9zYW5kYm94L2dRRE5DcFc5QkdpaWRvMWdHV0lwcW4taW1hZ2VzXzE3OTU0NTg0NDUwMDlfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTm9ZWEpsTDI1dmRtRndiR0Y1WDNOMGNtVmhiV3hwZEM5cGJXRm5aWE12YzNSeVpXRnRiR2wwWDNCc1lYbGxjbDl3Y21Wa2FXTjBhVzl1LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5NzI5MjgwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQDAo8jhED3B5W2ZjUyznEtjB4tbAIb0kPHTDT~AXMVyhwIgdDD0VhA7A2PFQ8N80qYU2kUpESmQaiYVchR0ZnKNnQE_)

*Example app output. The displayed score belongs to this individual input, not to the model’s overall test accuracy.*

![Example segment-based offer idea and business testing note](https://private-us-east-1.manuscdn.com/sessionFile/QAOnFAiMQMuaV9NPtCpc2f/sandbox/gQDNCpW9BGiido1gGWIpqn-images_1795458445009_na1fn_L2hvbWUvdWJ1bnR1L3NoYXJlL25vdmFwbGF5X3N0cmVhbWxpdC9pbWFnZXMvc3RyZWFtbGl0X29mZmVyX3Rlc3Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUUFPbkZBaU1RTXVhVjlOUHRDcGMyZi9zYW5kYm94L2dRRE5DcFc5QkdpaWRvMWdHV0lwcW4taW1hZ2VzXzE3OTU0NTg0NDUwMDlfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTm9ZWEpsTDI1dmRtRndiR0Y1WDNOMGNtVmhiV3hwZEM5cGJXRm5aWE12YzNSeVpXRnRiR2wwWDI5bVptVnlYM1JsYzNRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5NzI5MjgwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQDO5vd9ZghQHNZONj2Qo4iD1A14MwlD72-JdE4bPYkm7QIgb53~RZnDIRRVE1Z4Fd4LI2HU6~tSnIp9AaVwwHesbBI_)

*The app connects a predicted segment to an offer idea and recommends measuring business impact against a holdout group.*

## Deployment

**Open the live Streamlit app:** [mobile-game-in-app-purchases.streamlit.app](https://mobile-game-in-app-purchases.streamlit.app/)

The public app is deployed through Streamlit Community Cloud from this GitHub repository. Its Streamlit entry point and saved model are in [`app/app.py`](app/app.py) and [`app/model.joblib`](app/model.joblib). The `app/requirements.txt` file records the runtime dependencies. The app loads the saved preprocessing and Gini Decision Tree pipeline when it starts; it does not retrain the model for every visitor.

## Repository layout

The main project files are arranged in the same order as the repository:

```text
.
├── .devcontainer/
│   └── devcontainer.json        # Development-container settings
├── app/
│   ├── .gitignore
│   ├── README.md                # App setup and deployment notes
│   ├── app.py                   # Streamlit interface
│   ├── main.py                  # Supporting app script
│   ├── model.joblib             # Saved preprocessing + Gini Decision Tree
│   ├── requirements.txt         # App dependencies
│   └── train_model.py           # Rebuilds the saved model
├── data/
│   ├── 01_original_kaggle_data.csv
│   ├── 02_expanded_dirty_data.zip
│   ├── 03_clean_data.zip
│   └── 04_mobile_eda_features.zip
├── docs/
│   ├── 01_data_dictionary.md
│   └── 02_dataset_search_evidence.pdf
├── images/                      # EDA figures and app screenshots
├── notebooks/                   # Data, cleaning, EDA and modeling notebooks
├── LICENSE
└── README.md
```

The notebooks are numbered by project stage: data generation, cleaning, EDA and feature engineering, then model development and comparison. The `.devcontainer/devcontainer.json` file sets up a consistent development environment in compatible editors; it is not required for the deployed app.


## Business use and limitations

Use the predicted segment to define a small offer pilot, then compare it with a randomly assigned holdout group. Evaluate **incremental IAP revenue, contribution margin, conversion, and retention** before expanding a campaign. The prediction alone cannot show that the offer caused a revenue increase. The data are synthetic, so the reported results and learned patterns should be checked against representative real-player data before they guide live campaigns.

## Team

| Member | Student ID / GitHub | Main contribution |
|---|---|---|
| Hessa Khalfan | H00535166 · [@HesKal](https://github.com/HesKal) | Business problem formulation; dataset preparation, validation, and data-quality visualisations; exploratory data analysis (EDA) and feature engineering; and Streamlit app development. |
| Nourah Abdulla | H00535834 · [@NourahAlghfeli](https://github.com/NourahAlghfeli) | Data understanding and cleaning; quality checks and cleaned-dataset review. |
| Maryam Ali Bin Hamdah | H00535836 · [@MaryamBinHamdah](https://github.com/MaryamBinHamdah) | Data cleaning, dataset search and selection; source documentation and data dictionary. |

## AI use statement

Gemini and Copilot  was used to support coding, debugging, and code organisation. The team reviewed and tested the AI-assisted code and checked the reported results and project decisions before including them in the repository.

## License and attribution

The team’s original code is licensed under the [MIT License](LICENSE). The Kaggle dataset and other third-party materials remain subject to their respective licences and terms.

<div align="center">

*CDS4003 · NovaPlay Studios player intelligence project*

</div>
