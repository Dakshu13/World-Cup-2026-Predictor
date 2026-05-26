import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Connect to database
conn = sqlite3.connect('wc2026.db')

# Load data
teams = pd.read_sql_query('SELECT * FROM teams', conn)
players = pd.read_sql_query('SELECT * FROM players', conn)
conn.close()

# ---- Feature Engineering ----

# 1 — Form score (convert WWWDL to number)
def form_to_score(form):
    score = 0
    for result in form:
        if result == 'W': score += 3
        elif result == 'D': score += 1
        elif result == 'L': score += 0
    return score

teams['form_score'] = teams['form_last5'].apply(form_to_score)

# 2 — Goals per match in qualifiers
teams['goals_per_match'] = (teams['qualifier_goals_scored'] /
                             teams['qualifier_matches_played'])

# 3 — Goals conceded per match
teams['conceded_per_match'] = (teams['qualifier_goals_conceded'] /
                                teams['qualifier_matches_played'])

# 4 — Win rate in qualifiers
teams['win_rate'] = (teams['qualifier_wins'] /
                     teams['qualifier_matches_played'])

# 5 — Average player stats per team
player_stats = players.groupby('team').agg(
    avg_form_rating=('current_form_rating', 'mean'),
    avg_world_class=('world_class_rating', 'mean'),
    total_goals=('club_goals_2024_25', 'sum'),
    total_assists=('club_assists_2024_25', 'sum'),
    injury_risk_count=('injury_risk', lambda x:
                       (x == 'High').sum())
).reset_index()

# 6 — Merge teams and player stats
df = teams.merge(player_stats, on='team', how='left')

# 7 — Fill NaN with median for teams without players
numeric_cols = ['avg_form_rating', 'avg_world_class',
                'total_goals', 'total_assists']
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 8 — FIFA ranking score (invert so lower rank = higher score)
df['ranking_score'] = (df['fifa_ranking'].max() -
                        df['fifa_ranking'] + 1)

# 9 — Experience score
df['experience_score'] = (df['previous_wc_appearances'] * 2 +
                           df['best_wc_result'].map({
                               'Winner': 10,
                               'Runner Up': 8,
                               '3rd Place': 6,
                               '4th Place': 5,
                               'Quarter Final': 4,
                               'Round of 16': 3,
                               'Group Stage': 2,
                               'Debut': 1
                           }))

# 10 — TEAM STRENGTH SCORE (combining everything)
df['team_strength_score'] = (
    df['ranking_score'] * 0.20 +
    df['qualifier_points'] * 0.20 +
    df['form_score'] * 0.15 +
    df['win_rate'] * 10 * 0.15 +
    df['goals_per_match'] * 5 * 0.10 +
    df['avg_world_class'] * 0.10 +
    df['avg_form_rating'] * 0.05 +
    df['experience_score'] * 0.05
)

# Normalise to 0-100
df['team_strength_score'] = (
    (df['team_strength_score'] -
     df['team_strength_score'].min()) /
    (df['team_strength_score'].max() -
     df['team_strength_score'].min()) * 100
)

# Save for next stage
df.to_csv('team_features.csv', index=False)

# ---- Visualisations ----
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('FIFA World Cup 2026 — Feature Engineering',
             fontsize=16, fontweight='bold')

# Chart 1 — Top 20 teams by strength score
top20 = df.nlargest(20, 'team_strength_score')
colors = ['gold' if t in ['Argentina','France','Brazil']
          else 'royalblue' for t in top20['team']]
axes[0].barh(top20['team'][::-1],
             top20['team_strength_score'][::-1],
             color=colors[::-1])
axes[0].set_title('Top 20 Teams by Strength Score',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Team Strength Score (0-100)')

# Chart 2 — Feature correlation heatmap
features = ['ranking_score', 'qualifier_points',
            'form_score', 'win_rate', 'goals_per_match',
            'avg_world_class', 'experience_score',
            'team_strength_score']
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Feature Correlation Heatmap',
                   fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_engineering.png', dpi=150)
plt.show()
print("\nTop 10 Teams by Strength Score:")
print(df[['team','team_strength_score']]
      .nlargest(10, 'team_strength_score')
      .to_string(index=False))