import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('wc2026.db')

# Load CSVs
teams = pd.read_csv('wc2026_teams.csv')
players = pd.read_csv('wc2026_players.csv')
h2h = pd.read_csv('wc2026_head_to_head.csv')

# Save to SQL tables
teams.to_sql('teams', conn, if_exists='replace', index=False)
players.to_sql('players', conn, if_exists='replace', index=False)
h2h.to_sql('head_to_head', conn, if_exists='replace', index=False)

# Test SQL join query
query = '''
SELECT t.team, t.fifa_ranking, t.qualifier_points,
       t.confederation, t.form_last5,
       COUNT(p.player_name) as squad_size,
       ROUND(AVG(p.world_class_rating),2) as avg_world_class_rating,
       ROUND(AVG(p.current_form_rating),2) as avg_player_form,
       SUM(p.club_goals_2024_25) as total_squad_goals,
       SUM(p.club_assists_2024_25) as total_squad_assists
FROM teams t
LEFT JOIN players p ON t.team = p.team
GROUP BY t.team
ORDER BY t.fifa_ranking ASC
'''

result = pd.read_sql_query(query, conn)
print(result.head(10))
print(f"\nDatabase created with {len(teams)} teams and {len(players)} players!")
conn.close()