"""Auditoria em blocos do IMDb e cruzamento exploratório sem alterar a regressão."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def read_chunks(path, size):
    return pd.read_csv(path, sep='\t', compression='gzip', dtype='string',
        keep_default_na=False, na_values=['\\N', ''], quoting=csv.QUOTE_NONE,
        chunksize=size, on_bad_lines='error')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tmdb', type=Path)
    parser.add_argument('--chunksize', type=int, default=100000)
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error('--chunksize precisa ser positivo')
    root = args.project_root
    out, processed = root/'results/imdb', root/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    tmdb_path = args.tmdb or root/'data/raw/TMDB_all_movies.csv'
    if not tmdb_path.exists() and args.tmdb is None:
        tmdb_path = root/'TMDB_all_movies.csv'
    files = {'basics':root/'data/raw/title.basics.tsv.gz',
             'ratings':root/'data/raw/title.ratings.tsv.gz'}
    for path in [tmdb_path, *files.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    protected = [p for p in [root/'results/test_metrics.csv', root/'results/audit.json',
                             root/'results/test_predictions.csv', root/'results/model.joblib'] if p.exists()]
    before_hashes = {p.name:digest(p) for p in protected}
    tmdb = pd.read_csv(tmdb_path, usecols=['id','imdb_id','status','release_date','revenue'],
                       dtype={'imdb_id':'string'}, low_memory=False)
    target_ids = set(tmdb.imdb_id.dropna().str.strip())
    types, missing_before, missing_after, invalid = Counter(), Counter(), Counter(), Counter()
    per_type, matches, ids = [], [], []
    total, future_years = 0, 0
    fields = ['tconst','titleType','primaryTitle','originalTitle','isAdult',
              'startYear','endYear','runtimeMinutes','genres']
    for i, data in enumerate(read_chunks(files['basics'], args.chunksize)):
        if set(fields) - set(data):
            raise ValueError('Esquema title.basics inesperado')
        total += len(data)
        missing_before.update(data[fields].isna().sum().to_dict())
        for field in fields:
            data[field] = data[field].str.strip().replace('', pd.NA)
        type_key = data.titleType.fillna('unknown')
        types.update(type_key.value_counts().to_dict())
        valid_id = data.tconst.str.fullmatch(r'tt\d{7,}', na=False)
        invalid['invalid_tconst'] += int((~valid_id).sum())
        # Apenas os sufixos inteiros são acumulados para auditar chaves, não as linhas completas.
        numeric_ids = pd.to_numeric(data.loc[valid_id,'tconst'].str[2:], errors='raise').to_numpy(dtype=np.int64)
        ids.append(numeric_ids)
        for field in ['startYear','endYear','runtimeMinutes','isAdult']:
            original = data[field]
            converted = pd.to_numeric(original, errors='coerce').astype(float)
            bad = original.notna() & converted.isna()
            if field == 'isAdult':
                bad |= converted.notna() & ~converted.isin([0,1])
            else:
                bad |= converted.notna() & ((converted <= 0) | (converted % 1 != 0))
            invalid[field] += int(bad.sum())
            converted.loc[bad] = np.nan
            data[field] = converted
        wrong_end = data.endYear.notna() & data.startYear.notna() & data.endYear.lt(data.startYear)
        invalid['end_before_start'] += int(wrong_end.sum())
        data.loc[wrong_end,'endYear'] = np.nan
        series = type_key.isin(['tvSeries','tvMiniSeries'])
        non_applicable_end = (~series) & data.endYear.notna()
        invalid['endYear_on_non_series'] += int(non_applicable_end.sum())
        data.loc[~series,'endYear'] = np.nan
        future_years += int(data.startYear.gt(2026).sum())
        data.loc[~valid_id,'tconst'] = pd.NA
        missing_after.update(data[fields].isna().sum().to_dict())
        grouped = pd.DataFrame({'title_type':type_key, 'n':1,
            'missing_runtime':data.runtimeMinutes.isna().astype(int),
            'missing_start_year':data.startYear.isna().astype(int),
            'missing_genres':data.genres.isna().astype(int)}).groupby('title_type').sum()
        per_type.append(grouped)
        matches.append(data.loc[data.tconst.isin(target_ids),
            ['tconst','titleType','startYear','endYear','runtimeMinutes','genres','isAdult']].copy())
        if (i+1) % 20 == 0:
            print(f'IMDb basics: {total:,} títulos tratados', flush=True)
    all_ids = np.concatenate(ids) if ids else np.empty(0,dtype=np.int64)
    duplicate_keys = len(all_ids)-len(np.unique(all_ids))
    del ids, all_ids
    metadata = pd.concat(matches, ignore_index=True)
    matched_duplicates = int(metadata.tconst.duplicated().sum())
    metadata = metadata.drop_duplicates('tconst', keep='first')
    del matches
    by_type = pd.concat(per_type).groupby(level=0).sum().sort_values('n',ascending=False)
    by_type.to_csv(out/'title_types.csv')
    missing = pd.DataFrame([{'field':f,'missing_before':int(missing_before[f]),
        'missing_after':int(missing_after[f]),'missing_after_fraction':missing_after[f]/total} for f in fields])
    missing.to_csv(out/'missing_values.csv',index=False)

    rating_total, rating_invalid, rating_missing_before, rating_missing_after = 0, Counter(), Counter(), Counter()
    rating_matches, rating_ids = [], []
    for data in read_chunks(files['ratings'], args.chunksize):
        rating_total += len(data)
        rating_missing_before.update(data.isna().sum().to_dict())
        valid_id = data.tconst.str.fullmatch(r'tt\d{7,}',na=False)
        rating_invalid['invalid_tconst'] += int((~valid_id).sum())
        rating_ids.append(pd.to_numeric(data.loc[valid_id,'tconst'].str[2:],errors='raise').to_numpy(dtype=np.int64))
        for field in ['averageRating','numVotes']:
            original = data[field]
            converted = pd.to_numeric(original,errors='coerce').astype(float)
            bad = original.notna() & converted.isna()
            if field == 'averageRating':
                bad |= converted.notna() & ~converted.between(1,10)
            else:
                bad |= converted.notna() & ((converted <= 0) | (converted % 1 != 0))
            rating_invalid[field] += int(bad.sum())
            converted.loc[bad] = np.nan
            data[field] = converted
        data.loc[~valid_id,'tconst'] = pd.NA
        rating_missing_after.update(data.isna().sum().to_dict())
        rating_matches.append(data.loc[data.tconst.isin(target_ids)].copy())
    rating_key_array = np.concatenate(rating_ids)
    rating_duplicates = len(rating_key_array)-len(np.unique(rating_key_array))
    ratings = pd.concat(rating_matches,ignore_index=True).drop_duplicates('tconst')
    metadata = metadata.merge(ratings,on='tconst',how='left',validate='one_to_one')
    metadata = metadata.rename(columns={c:'imdb_'+c for c in metadata if c != 'tconst'})
    joined = tmdb.merge(metadata,left_on='imdb_id',right_on='tconst',how='left',
                        validate='many_to_one',indicator='imdb_match')
    joined.imdb_titleType.fillna('unmatched').value_counts().rename_axis('title_type').to_csv(out/'tmdb_overlap_types.csv')
    movie_reference = joined.loc[joined.imdb_titleType.eq('movie')].copy()
    movie_reference.to_csv(processed/'tmdb_imdb_movies_reference.csv',index=False)
    dates = pd.to_datetime(joined.release_date,errors='coerce')
    model_mask = joined.status.eq('Released') & dates.notna() & dates.le(pd.Timestamp('2025-10-18')) & joined.revenue.gt(0)
    model_coverage = joined.loc[model_mask,'imdb_titleType'].fillna('unmatched').value_counts()
    model_coverage.rename_axis('title_type').to_csv(out/'model_sample_coverage.csv')
    preserved = {p.name: digest(p) == before_hashes[p.name] for p in protected}
    if not all(preserved.values()):
        raise AssertionError('A etapa exploratória alterou resultados protegidos da modelagem')
    audit = {
        'processed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'chunksize':args.chunksize, 'basics_rows':total, 'ratings_rows':rating_total,
        'basic_types':dict(types), 'duplicate_numeric_imdb_keys':int(duplicate_keys),
        'matched_duplicate_tconst_rows':matched_duplicates, 'invalid_basics':dict(invalid),
        'future_start_year_rows_flagged_not_removed':future_years,
        'ratings_invalid':dict(rating_invalid), 'ratings_duplicate_numeric_keys':int(rating_duplicates),
        'ratings_missing_before':dict(rating_missing_before), 'ratings_missing_after':dict(rating_missing_after),
        'tmdb_rows':len(tmdb), 'tmdb_rows_with_imdb_id':int(tmdb.imdb_id.notna().sum()),
        'tmdb_rows_with_basic_match':int(joined.imdb_match.eq('both').sum()),
        'tmdb_matched_movie_reference_rows':len(movie_reference),
        'tmdb_rows_with_current_imdb_rating':int(joined.imdb_averageRating.notna().sum()),
        'original_model_sample_rows':int(model_mask.sum()),
        'original_model_sample_type_coverage':model_coverage.to_dict(),
        'model_outputs_preserved':preserved,
        'sources':{name:{'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size}
                   for name,path in files.items()},
        'sample_policy':'Separate source profiles; movie-only reference join is exploratory and is not fed into training.',
        'end_year_policy':'Not imputed: structural absence for titles other than tvSeries/tvMiniSeries.',
        'date_policy':'Future startYear values are flagged, not converted to missing.',
        'key_audit_policy':'Valid tt identifiers converted to integer suffixes; overlapping matched tconst rows deduplicated explicitly.'}
    (out/'audit_imdb.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4.6))
    by_type['n'].head(9).sort_values().plot.barh(ax=ax,color='#497B9B')
    ax.set_xlabel('Quantidade de títulos'); ax.set_ylabel('Tipo de título IMDb')
    ax.ticklabel_format(axis='x',style='plain')
    fig.tight_layout(); fig.savefig(out/'title_types.png',dpi=180); plt.close(fig)
    print(json.dumps(audit,indent=2,ensure_ascii=True),flush=True)

if __name__ == '__main__':
    main()
