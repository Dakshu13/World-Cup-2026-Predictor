import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to database
conn = sqlite3.connect('wc2026.db')

# Load data
teams = pd.read_sql_query('SELECT * FROM teams', conn)
players = pd.read_sql_query('SELECT * FROM players', conn)

# Feature engineering
teams['goals_per_match'] = (teams['qualifier_goals_scored'] /
                             teams['qualifier_matches_played'])

conn.close()

# Set style
sns.set_style("darkgrid")
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('FIFA World Cup 2026 — Exploratory Data Analysis',
             fontsize=18, fontweight='bold')

# Chart 1 — Top 10 teams by FIFA ranking
top10 = teams.nsmallest(10, 'fifa_ranking')
axes[0,0].barh(top10['team'][::-1],
               top10['qualifier_points'][::-1],
               color=sns.color_palette("husl", 10))
axes[0,0].set_title('Top 10 Teams — Qualifier Points',
                     fontsize=13, fontweight='bold')
axes[0,0].set_xlabel('Qualifier Points')

# Chart 2 — Goals scored vs conceded scatter
colors = {'UEFA':'blue','CONMEBOL':'green',
          'CONCACAF':'orange','CAF':'red',
          'AFC':'purple'}
for conf in teams['confederation'].unique():
    subset = teams[teams['confederation'] == conf]
    axes[0,1].scatter(subset['qualifier_goals_scored'],
                      subset['qualifier_goals_conceded'],
                      label=conf, alpha=0.7, s=80,
                      color=colors.get(conf,'grey'))
axes[0,1].set_title('Goals Scored vs Goals Conceded by Confederation',
                     fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Goals Scored')
axes[0,1].set_ylabel('Goals Conceded')
axes[0,1].legend()

# Chart 3 — Average world class rating by confederation
avg_rating = (players.merge(
    teams[['team','confederation']], on='team', how='left')
    .groupby('confederation')['world_class_rating']
    .mean()
    .sort_values(ascending=False)
    .reset_index())
axes[1,0].bar(avg_rating['confederation'],
              avg_rating['world_class_rating'],
              color=sns.color_palette("husl", len(avg_rating)))
axes[1,0].set_title('Average World Class Player Rating by Confederation',
                     fontsize=13, fontweight='bold')
axes[1,0].set_xlabel('Confederation')
axes[1,0].set_ylabel('Average Rating')
axes[1,0].set_ylim(7, 9.5)

# Chart 4 — Top 10 players by form rating
top_players = (players.nlargest(10, 'current_form_rating')
               [['player_name','team','current_form_rating',
                 'position']])
bars = axes[1,1].barh(
    top_players['player_name'][::-1],
    top_players['current_form_rating'][::-1],
    color=sns.color_palette("husl", 10))
axes[1,1].set_title('Top 10 Players by Current Form Rating',
                     fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Form Rating')
axes[1,1].set_xlim(8.5, 10)
for i, (_, row) in enumerate(top_players[::-1].iterrows()):
    axes[1,1].text(row['current_form_rating'] + 0.01,
                   i, f"{row['team']}",
                   va='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150)
plt.show()
print("EDA completed!")