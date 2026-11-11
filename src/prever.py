"""Consulta local ao pipeline experimental salvo em results/model.joblib."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--budget', type=float, default=0)
parser.add_argument('--runtime', type=float, default=0)
parser.add_argument('--genres', default='', help='Generos em ingles separados por virgula')
parser.add_argument('--language', required=True)
parser.add_argument('--year', required=True, type=int)
parser.add_argument('--month', required=True, type=int, choices=range(1,13))
args = parser.parse_args()
audit = json.loads((root/'results/audit.json').read_text(encoding='utf-8'))
features = audit['features']
row = {f: 0 for f in features}
row.update(log_budget=np.log1p(args.budget) if args.budget > 0 else np.nan,
           log_runtime=np.log1p(args.runtime) if args.runtime > 0 else np.nan,
           year=args.year, month=args.month, original_language=args.language)
for genre in args.genres.split(','):
    key = 'genre_' + genre.strip()
    if key in row:
        row[key] = 1
model = joblib.load(root/'results/model.joblib')
prediction = float(model.predict(pd.DataFrame([row], columns=features))[0])
print(json.dumps({'estimated_revenue_nominal': prediction,
                  'status': 'experimental_exploratory_model'}, indent=2))
