"""Experimento reproduzível; execute na raiz do projeto."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/initial'
SEED = 42

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def score(model, x, y):
    pred = model.predict(x)
    return {'mae': mean_absolute_error(y, pred),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': r2_score(y, pred)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmdb_path = ROOT / 'data/raw/TMDB_all_movies.csv'
    if not tmdb_path.exists():
        tmdb_path = ROOT / 'TMDB_all_movies.csv'
    for path in (tmdb_path,):
        if not path.exists():
            raise FileNotFoundError(f'Baixe o arquivo indicado no README: {path}')
    columns = ['id', 'imdb_id', 'release_date', 'runtime', 'budget', 'genres',
               'original_language', 'revenue', 'status', 'popularity',
               'vote_average', 'vote_count', 'imdb_rating']
    # Evita carregar sinopses, elenco e outros textos extensos sem uso experimental.
    data = pd.read_csv(tmdb_path, usecols=lambda c: c in columns, low_memory=False)
    required = {'id', 'imdb_id', 'release_date', 'runtime', 'budget',
                'genres', 'original_language', 'revenue'}
    if required - set(data):
        raise ValueError(f'Colunas ausentes: {sorted(required - set(data))}')
    audit = {'raw_rows': len(data), 'tmdb_duplicate_ids': int(data.id.duplicated().sum())}
    audit['raw_missing_fraction'] = data.isna().mean().to_dict()
    audit['raw_positive_revenue'] = int(pd.to_numeric(data.revenue, errors='coerce').gt(0).sum())
    data = data.drop_duplicates('id').copy()
    if 'status' in data:
        data = data.loc[data.status.eq('Released')].copy()
    audit['released_rows'] = len(data)
    data['revenue'] = pd.to_numeric(data.revenue, errors='coerce')
    data['date'] = pd.to_datetime(data.release_date, errors='coerce')
    snapshot_date = pd.Timestamp('2026-10-18')
    # Exclui lançamentos recentes cuja receita pode ainda estar em formação.
    data = data.loc[data.date.notna()
                    & (data.date <= snapshot_date - pd.DateOffset(years=1))
                    & data.revenue.gt(0)].copy()
    audit['positive_revenue_rows'] = len(data)
    data.groupby(data.date.dt.year).revenue.agg(['count', 'median', 'mean']).to_csv(OUT / 'revenue_by_year.csv')
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    for col in ['budget', 'runtime']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data.loc[data[col] <= 0, col] = np.nan
    audit['missing_fraction'] = data[list(required)].isna().mean().to_dict()
    data['log_budget'] = np.log1p(data.budget)
    data['target'] = data.revenue
    genres = data.genres.fillna('').str.get_dummies(sep=',')
    genres.columns = ['genre_' + c.strip() for c in genres.columns]
    genres = genres.T.groupby(level=0).max().T
    data = pd.concat([data, genres], axis=1)
    genre_summary = []
    for genre in genres:
        subset = data.loc[data[genre].eq(1)]
        genre_summary.append({'genre': genre.removeprefix('genre_'), 'n': len(subset),
                              'median_revenue': subset.revenue.median()})
    pd.DataFrame(genre_summary).to_csv(OUT / 'revenue_by_genre.csv', index=False)
    years = sorted(data.year.unique())
    if len(years) < 10:
        raise ValueError('Poucos anos para a separação temporal proposta.')
    # Cortes por quantidade acumulada, sem dividir o mesmo ano entre conjuntos.
    cumulative = data.groupby('year').size().sort_index().cumsum() / len(data)
    train_end = int(cumulative.index[cumulative.ge(.70)][0])
    val_end = int(cumulative.index[cumulative.ge(.85)][0])
    train = data.loc[data.year <= train_end]
    val = data.loc[(data.year > train_end) & (data.year <= val_end)]
    test = data.loc[data.year > val_end]
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        if len(part) < 2:
            raise ValueError(f'{name}: menos de dois filmes; reveja cortes.')
        audit[name] = {'n': len(part), 'first_year': int(part.year.min()),
                       'last_year': int(part.year.max()), 'median_revenue': float(part.target.median())}
    numeric = ['log_budget', 'runtime', 'year', 'month'] + list(genres.columns)
    categorical = ['original_language']
    features = numeric + categorical
    def pipeline(model):
        numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                                 ('scale', StandardScaler())])
        category_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))])
        return Pipeline([('prepare', ColumnTransformer([
            ('numeric', numeric_pipe, numeric), ('category', category_pipe, categorical)])),
            ('model', TransformedTargetRegressor(regressor=model,
                                                  func=np.log1p, inverse_func=np.expm1))])
    models = {'dummy': DummyRegressor(strategy='median'),
              'ridge': Ridge(alpha=10, solver='lsqr'),
              'forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                  n_jobs=-1, random_state=SEED)}
    trained, validation = {}, []
    for name, estimator in models.items():
        fitted = pipeline(estimator).fit(train[features], train.target)
        trained[name] = fitted
        validation.append({'model': name, **score(fitted, val[features], val.target)})
    table = pd.DataFrame(validation).sort_values('mae', ascending=True)
    table.to_csv(OUT / 'validation_metrics.csv', index=False)
    best = str(table.iloc[0]['model'])
    development = pd.concat([train, val])
    final = pipeline(models[best]).fit(development[features], development.target)
    test_metrics = {'model': best, **score(final, test[features], test.target)}
    pd.DataFrame([test_metrics]).to_csv(OUT / 'test_metrics.csv', index=False)
    joblib.dump(final, OUT / 'model.joblib')
    sns.histplot(np.log1p(data.revenue), bins=40)
    plt.xlabel('log(1 + receita)'); plt.ylabel('Filmes'); plt.tight_layout()
    plt.savefig(OUT / 'receita.png', dpi=180); plt.close()
    prediction = final.predict(test[features])
    pd.DataFrame({'id': test.id, 'year': test.year, 'observed_revenue': test.target,
                  'predicted_revenue': prediction}).to_csv(OUT / 'test_predictions.csv', index=False)
    plt.scatter(test.target, prediction, alpha=.25, s=10)
    plt.xscale('log'); plt.yscale('symlog'); plt.xlabel('Receita observada')
    plt.ylabel('Receita estimada'); plt.tight_layout()
    plt.savefig(OUT / 'predicoes.png', dpi=180); plt.close()
    exploratory = [c for c in ['budget', 'revenue', 'popularity', 'vote_average',
                              'vote_count', 'runtime', 'imdb_rating'] if c in data]
    corr = data[exploratory].apply(pd.to_numeric, errors='coerce').corr(method='spearman')
    corr.to_csv(OUT / 'correlations_spearman.csv')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1)
    plt.tight_layout(); plt.savefig(OUT / 'correlacoes.png', dpi=180); plt.close()
    positive_budget = data.loc[data.budget.gt(0)]
    plt.scatter(positive_budget.budget, positive_budget.revenue, alpha=.15, s=8)
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Orçamento nominal')
    plt.ylabel('Receita nominal'); plt.tight_layout()
    plt.savefig(OUT / 'orcamento_receita.png', dpi=180); plt.close()
    audit.update({'target': 'revenue', 'features': features,
                  'seed': SEED, 'snapshot_date': str(snapshot_date.date()),
                  'files': {p.name: sha256(p) for p in [tmdb_path]},
                  'python': platform.python_version(),
                  'packages': {p: importlib.metadata.version(p) for p in
                               ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']}})
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(table.to_string(index=False)); print('Teste final:', test_metrics)

if __name__ == '__main__':
    main()

