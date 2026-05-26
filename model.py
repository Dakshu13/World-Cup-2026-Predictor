import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load feature engineered data
df = pd.read_csv('team_features.csv')

# ---- Prepare features ----
features = [
    'ranking_score', 'qualifier_points', 'form_score',
    'win_rate', 'goals_per_match', 'avg_world_class',
    'avg_form_rating', 'experience_score'
]

X = df[features].fillna(0)

# Create target — top 16 teams = 1, rest = 0
df['target'] = (df['team_strength_score'] >= 
                df['team_strength_score'].nlargest(16).min()
                ).astype(int)

y = df['target']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Hyperparameter Tuning ----
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, 
                           cv=3, scoring='f1',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# ---- Evaluation ----
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---- Predict all teams ----
df['win_probability'] = best_model.predict_proba(X)[:, 1]
df['win_probability'] = (df['win_probability'] * 100).round(1)

# ---- Visualisations ----
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('FIFA World Cup 2026 — Prediction Model',
             fontsize=16, fontweight='bold')

# Chart 1 — Feature Importance
importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(importance['feature'], 
             importance['importance'],
             color=sns.color_palette("husl", len(features)))
axes[0].set_title('Feature Importance — Random Forest',
                   fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance Score')

# Chart 2 — Top 10 predicted winners
top10 = df.nlargest(10, 'win_probability')
colors = ['gold' if t in ['Argentina', 'Brazil', 'France']
          else 'royalblue' for t in top10['team']]
axes[1].barh(top10['team'][::-1],
             top10['win_probability'][::-1],
             color=colors[::-1])
axes[1].set_title('Top 10 Predicted Winners — Win Probability %',
                   fontsize=12, fontweight='bold')
axes[1].set_xlabel('Win Probability (%)')

# Chart 3 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            ax=axes[2],
            xticklabels=['Not Top 16', 'Top 16'],
            yticklabels=['Not Top 16', 'Top 16'])
axes[2].set_title('Confusion Matrix',
                   fontsize=12, fontweight='bold')
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=150)
plt.show()

# ---- Final Summary ----
print("\n" + "="*50)
print("FIFA WORLD CUP 2026 — TOP 10 PREDICTED WINNERS")
print("="*50)
print(df[['team', 'win_probability', 'team_strength_score']]
      .nlargest(10, 'win_probability')
      .to_string(index=False))