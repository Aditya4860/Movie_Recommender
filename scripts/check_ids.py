import pandas as pd, warnings
warnings.filterwarnings('ignore')
m = pd.read_csv('data/tmdb_5000_movies.csv')
c = pd.read_csv('data/tmdb_5000_credits.csv')
merged = m.merge(c[['title','cast','crew']], on='title', how='inner')
merged_set = set(merged['title'].tolist())
check = ['Monsters, Inc.','The Incredibles','WALL-E','Up','Finding Nemo','Toy Story 2','Toy Story 3']
for t in check:
    row = m[m['title']==t]
    mid = int(row.iloc[0]['id']) if not row.empty else 'N/A'
    exists = 'YES' if t in merged_set else 'NO'
    print(f'[{exists}] id={mid} {t}')
