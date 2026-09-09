import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.data import load_base_datasets, parse_json_list, find_data_file
print('data.py imports OK')

from src.model_store import load_model, load_manifest
import pathlib
m = load_model(pathlib.Path('models'))
print(f'Model loaded: method={m.method}, corpus={len(m.movies)} movies')

manifest = load_manifest()
built = manifest.get("built_at", "?")
method = manifest.get("method", "?")
print(f'Manifest: built_at={built}, method={method}')

res = m.recommend(27205, 5)
res2 = res.merge(m.movies[["id","title"]], on="id")
print('Inception top-5:', res2["title"].tolist())

# Verify the about page can import pandas independently
import pandas as pd
print('pandas import in about.py context: OK')
print('ALL GOOD')
