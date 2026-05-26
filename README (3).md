# FIFA World Cup 2026 — Match Outcome Predictor 🏆

A complete end-to-end Data Science pipeline that predicts the top 16 qualifying teams and potential winners of the FIFA World Cup 2026 using machine learning.

## Project Overview

This project builds a Random Forest classification model trained on team statistics, player ratings, qualifier performance, and historical World Cup data to predict which teams are most likely to advance and win the 2026 World Cup.

## Pipeline Architecture

```
Raw CSV Data (Teams + Players + Head-to-Head)
        ↓
SQLite Database (setup_db.py)
        ↓
Exploratory Data Analysis (eda.py)
        ↓
Feature Engineering (features.py)
        ↓
Random Forest Model + GridSearchCV (model.py)
        ↓
Predictions + Visualisations
```

## Dataset

Three datasets used in this project:

| File | Description | Rows |
|---|---|---|
| `wc2026_teams.csv` | FIFA rankings, qualifier stats, form, confederation | 58 teams |
| `wc2026_players.csv` | Player ratings, club form, goals, assists, injury risk | 69 players |
| `wc2026_head_to_head.csv` | Historical head-to-head match records | - |

**Source:** Custom dataset built using FIFA rankings, BTS qualifier data, and club performance statistics (2024-25 season).

## Features Engineered

| Feature | Description |
|---|---|
| `ranking_score` | Inverted FIFA ranking (higher = better) |
| `qualifier_points` | Points earned in World Cup qualifiers |
| `form_score` | Last 5 matches converted to numeric (W=3, D=1, L=0) |
| `win_rate` | Qualifier wins / matches played |
| `goals_per_match` | Qualifier goals scored / matches played |
| `avg_world_class` | Average world class rating of squad players |
| `avg_form_rating` | Average current form rating of squad players |
| `experience_score` | Weighted score from past WC appearances and best result |
| `team_strength_score` | Composite score combining all features (normalised 0-100) |

## Model

- **Algorithm:** Random Forest Classifier
- **Tuning:** GridSearchCV with 3-fold cross validation
- **Target:** Binary classification — Top 16 (1) vs Not Top 16 (0)
- **Evaluation:** Precision, Recall, F1-Score, Confusion Matrix

### Hyperparameter Grid

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}
```

### Results

| Metric | Score |
|---|---|
| False Positives | 0 |
| False Negatives | 0 |
| Top feature | avg_world_class (0.32) |

## Top 10 Predicted Winners

| Rank | Team | Win Probability |
|---|---|---|
| 1 | Spain | ~100% |
| 2 | France | ~100% |
| 3 | Argentina | ~100% |
| 4 | Japan | ~98% |
| 5 | Portugal | ~97% |
| 6 | Brazil | ~96% |
| 7 | USA | ~95% |
| 8 | Morocco | ~94% |
| 9 | England | ~93% |
| 10 | South Korea | ~90% |

## Key Findings

- **Player quality (avg_world_class) is the strongest predictor** — individual talent matters more than historical ranking in tournament football
- **Recent form and win rate** are more predictive than FIFA ranking alone
- **UEFA teams** are defensively efficient — low goals conceded relative to goals scored
- **CONMEBOL teams** (Argentina, Brazil) dominate overall strength scores
- **Experience score** had the lowest feature importance — suggesting past World Cup history is less predictive than current form

## Limitations & Future Improvements

- Player injury and fitness data on matchday not included
- Player mental health and training feedback not available
- Head-to-head data not yet fully integrated into the model
- Small player dataset (69 players across 58 teams) — more complete squad data would improve accuracy
- Weather and venue conditions not considered
- Model trained on qualifier data — knockout stage dynamics differ significantly

## Project Structure

```
WorldCup2026Predictor/
│
├── wc2026_teams.csv          # Teams dataset
├── wc2026_players.csv        # Players dataset
├── wc2026_head_to_head.csv   # Head-to-head dataset
│
├── setup_db.py               # Stage 1 — Database setup and SQL joins
├── eda.py                    # Stage 2 — Exploratory Data Analysis
├── features.py               # Stage 3 — Feature Engineering
├── model.py                  # Stage 4 — Random Forest model + predictions
│
├── wc2026.db                 # SQLite database (auto-generated)
├── team_features.csv         # Engineered features (auto-generated)
│
├── eda_analysis.png          # EDA charts
├── feature_engineering.png   # Feature correlation heatmap
├── model_predictions.png     # Final prediction charts
│
└── README.md                 # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Dakshu13/WorldCup2026Predictor.git
cd WorldCup2026Predictor

# Install dependencies
pip install pandas matplotlib seaborn scikit-learn numpy
```

## Usage

Run each stage in order:

```bash
# Stage 1 — Set up database
python setup_db.py

# Stage 2 — Exploratory Data Analysis
python eda.py

# Stage 3 — Feature Engineering
python features.py

# Stage 4 — Prediction Model
python model.py
```

## Tech Stack

- **Python 3.x**
- **SQLite** — data storage and SQL joins
- **Pandas** — data manipulation
- **Scikit-learn** — Random Forest, GridSearchCV
- **Matplotlib / Seaborn** — visualisations
- **NumPy** — numerical operations

## Author

**Dakshina Moorthy Ramesh**  
MSc in Computing — Artificial Intelligence with NLP  
Dublin City University, Ireland  
[LinkedIn](https://www.linkedin.com/in/dakshina-moorthy-ramesh-22651221a/) | [GitHub](https://github.com/Dakshu13) | [Portfolio](https://qm7ljjkgpngi6.ok.kimi.link/)

---
*Built as part of Data Science interview preparation and MSc studies at DCU*
