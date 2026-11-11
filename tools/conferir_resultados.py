from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
root=Path(__file__).resolve().parents[1]
out=root/'results/review'
out.mkdir(parents=True,exist_ok=True)
raw=root/'TMDB_all_movies.csv'
if not raw.exists(): raw=root/'data/raw/TMDB_all_movies.csv'
d=pd.read_csv(raw,usecols=['id','imdb_id','release_date','status','revenue','runtime','budget'],low_memory=False)
d['date']=pd.to_datetime(d.release_date,errors='coerce')
d=d.loc[d.status.eq('Released') & d.date.notna() & d.date.le(pd.Timestamp('2025-10-18')) & d.revenue.gt(0)].copy()
d['part']=np.select([d.date.dt.year.le(2018),d.date.dt.year.le(2022)],['train','validation'],default='test')
with_id=d.dropna(subset=['imdb_id'])
groups=with_id.groupby('imdb_id').agg(n=('id','size'),partitions=('part','nunique'))
repeated=groups.loc[groups.n.gt(1)]
cross=groups.loc[groups.partitions.gt(1)]
pred=pd.read_csv(root/'results/test_predictions.csv')
m=pd.read_csv(root/'results/test_metrics.csv').iloc[0]
calculated={'mae':mean_absolute_error(pred.observed_revenue,pred.predicted_revenue),'rmse':np.sqrt(mean_squared_error(pred.observed_revenue,pred.predicted_revenue)),'r2':r2_score(pred.observed_revenue,pred.predicted_revenue)}
assert all(np.isclose(calculated[k],m[k]) for k in calculated)
a={'analytical_rows':len(d),'distinct_tmdb_ids':int(d.id.nunique()),'repeated_imdb_keys_in_sample':len(repeated),'rows_with_repeated_imdb_keys':int(repeated.n.sum()),'imdb_keys_crossing_temporal_partitions':len(cross),'rows_for_cross_partition_keys':int(cross.n.sum()),'cross_partition_rows_by_set':with_id.loc[with_id.imdb_id.isin(cross.index),'part'].value_counts().to_dict(),'test_metrics_verified':{k:float(v) for k,v in calculated.items()},'test_predictions_match_sample_ids':set(pred.id)==set(d.loc[d.part.eq('test'),'id'])}
(out/'checks.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
with_id.loc[with_id.imdb_id.isin(cross.index),['id','imdb_id','release_date','part']].to_csv(out/'cross_partition_ids.csv',index=False)
print(json.dumps(a,indent=2))

