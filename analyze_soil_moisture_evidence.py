#!/usr/bin/env python3
"""Post-hoc, physical-soil-clustered analysis of existing out-of-fold predictions.
Run: python analyze_soil_moisture_evidence.py --v4 soil_moisture_v4_scheduled_results.zip --v6 soil_moisture_v6_spectral_results.zip --out soil_evidence
Requires pandas numpy matplotlib; no GPU. All tests exploratory after multiple model trials.
"""
import argparse, zipfile, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

P=argparse.ArgumentParser();P.add_argument('--v4',required=True);P.add_argument('--v6',required=True);P.add_argument('--out',default='soil_evidence');P.add_argument('--bootstrap',type=int,default=3000);P.add_argument('--seed',type=int,default=2026)
a=P.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)

def load(zpath):
 with zipfile.ZipFile(zpath) as z:return pd.read_csv(z.open(next(x for x in z.namelist() if x.endswith('predictions.csv'))))
v4=load(a.v4);v6=load(a.v6)
# Check one prediction for each physical measurement, each model and validation; fail loudly otherwise.
for tag,frame in [('v4',v4),('v6',v6)]:
 assert not frame.duplicated(['validation','variant','row_index']).any(), f'duplicates {tag}'
 assert frame.groupby(['validation','variant'])['row_index'].nunique().eq(426).all(), f'missing obs {tag}'
for col in ['sample_id','site_id','water_content','true_Corg']:
 left=v4[(v4.validation=='sample_grouped')&(v4.variant=='combined')].set_index('row_index')
 right=v6[(v6.validation=='sample_grouped')&(v6.variant=='adaptive')].set_index('row_index')
 assert left.index.equals(right.index) or set(left.index)==set(right.index)
 assert np.all(left[col].sort_index().astype(str).values==right[col].sort_index().astype(str).values),f'nonmatched {col}'

frames=[]
for df,prefix in [(v4,'V4'),(v6,'V6')]:
 df=df.copy();df['model']=prefix+' '+df.variant
 frames.append(df)
allp=pd.concat(frames,ignore_index=True)
targets=['Corg','Ctot','Corg_Clay','clay','silt','sand','Canorg','DOC','Cmic','FeDi','FeOx']
variants=['V4 plsr_mixed','V4 plsr_t6','V4 combined','V4 consistency','V6 adaptive','V6 combined']

def metrics(x):
 d=[]
 for (validation,model),g in x[x.model.isin(variants)].groupby(['validation','model']):
  for t in targets:
   true=g['true_'+t].to_numpy();pred=g['pred_'+t].to_numpy()
   d.append(dict(validation=validation,model=model,target=t,n=len(g),n_soils=g.sample_id.nunique(),r2=r2_score(true,pred),mae=np.abs(true-pred).mean(),rmse=np.sqrt(np.mean((true-pred)**2))))
 return pd.DataFrame(d)
met=metrics(allp);met.to_csv(out/'model_metrics.csv',index=False)

# Soil-level paired bootstrap: resample physical soils, not observations. CI is descriptive (model selection already occurred).
rng=np.random.default_rng(a.seed)
comp=[('V4 plsr_t6','V4 plsr_mixed'),('V4 plsr_mixed','V4 combined'),('V4 plsr_mixed','V6 adaptive'),('V4 combined','V6 adaptive'),('V4 combined','V4 consistency')]
results=[];records=[]
for val in ['sample_grouped','site_grouped']:
 sub=allp[allp.validation==val]
 for left,right in comp:
  xa=sub[sub.model==left].set_index('row_index').sort_index();xb=sub[sub.model==right].set_index('row_index').sort_index()
  assert xa.index.equals(xb.index) and np.array_equal(xa.sample_id.values,xb.sample_id.values)
  cluster_col='site_id' if val=='site_grouped' else 'sample_id'
  clusters=xa[cluster_col].unique();idx=[np.flatnonzero(xa[cluster_col].values==s) for s in clusters]
  draws=rng.integers(0,len(clusters),size=(a.bootstrap,len(clusters)))
  # Precompute per-soil sums of abs err for bootstrap. Each soil is the independent resampling unit.
  for t in targets:
   erra=np.abs(xa['pred_'+t].values-xa['true_'+t].values)
   errb=np.abs(xb['pred_'+t].values-xb['true_'+t].values)
   n=np.array([len(j) for j in idx]);sum_a=np.array([erra[j].sum() for j in idx]);sum_b=np.array([errb[j].sum() for j in idx])
   delta=(sum_a.sum()-sum_b.sum())/n.sum()  # positive means right is better
   bs=((sum_a[draws]-sum_b[draws]).sum(axis=1)/n[draws].sum(axis=1))
   lo,hi=np.quantile(bs,[.025,.975]); results.append(dict(validation=val,target=t,model_a=left,model_b=right,mae_a=erra.mean(),mae_b=errb.mean(),mae_reduction_b_vs_a=delta,ci_low=lo,ci_high=hi,bootstrap_reps=a.bootstrap,unit=('sampling site' if val=='site_grouped' else 'physical soil'),exploratory=True))
   if t=='Corg':
    records.append((val,left,right,delta,lo,hi))
pairs=pd.DataFrame(results);pairs.to_csv(out/'paired_soil_bootstrap.csv',index=False)

# Moisture bins defined by observed WATER CONTENT, not drying time; all pooled oof held-out soil predictions.
moist=allp[allp.validation=='sample_grouped'].copy()
edges=np.unique(np.quantile(moist.drop_duplicates('row_index').water_content,[0,.25,.5,.75,1]))
assert len(edges)==5,'water-content quartile edges collapsed'
moist['moisture_quartile']=pd.cut(moist.water_content,bins=edges,include_lowest=True,labels=['Q1 driest','Q2','Q3','Q4 wettest'])
groups=[]
for (model,quart),g in moist[moist.model.isin(variants)].groupby(['model','moisture_quartile'],observed=True):
 for t in targets:
  groups.append(dict(model=model,quartile=str(quart),n=len(g),water_min=g.water_content.min(),water_max=g.water_content.max(),target=t,mae=np.abs(g['pred_'+t]-g['true_'+t]).mean()))
quart=pd.DataFrame(groups);quart.to_csv(out/'moisture_quartile_errors.csv',index=False)

# Within soil prediction SD and mean absolute error are reported together (otherwise stable but wrong predictions look good).
st=[]
for (validation,model,soil),g in allp[allp.model.isin(variants)].groupby(['validation','model','sample_id']):
 for t in targets:
  st.append(dict(validation=validation,model=model,sample_id=soil,target=t,n=len(g),pred_sd=g['pred_'+t].std(ddof=1),mae=np.abs(g['pred_'+t]-g['true_'+t]).mean(),mean_bias=np.mean(g['pred_'+t]-g['true_'+t]),wc_span=g.water_content.max()-g.water_content.min()))
st=pd.DataFrame(st);st.to_csv(out/'soil_trajectory_stability.csv',index=False)

# Fold agreement comparison also check fold assignment; not strictly required for paired sample errors but report it.
left=v4[(v4.validation=='sample_grouped')&(v4.variant=='combined')].set_index('row_index')
right=v6[(v6.validation=='sample_grouped')&(v6.variant=='adaptive')].set_index('row_index')
fold_agreement=float((left.fold.sort_index()==right.fold.sort_index()).mean())

# Figures (matplotlib automatic colors, user can recolor for journal).
plt.rcParams.update({'font.size':10,'figure.dpi':150})
sel=met[(met.validation=='sample_grouped')&(met.target.isin(['Corg','Ctot','clay','DOC','Cmic']))&(met.model.isin(['V4 plsr_mixed','V4 combined','V6 adaptive']))]
piv=sel.pivot(index='target',columns='model',values='r2').reindex(['Corg','Ctot','clay','DOC','Cmic'])
ax=piv.plot.bar(figsize=(9,4.2),rot=0);ax.set_ylabel('Out-of-fold R²');ax.set_xlabel('');ax.set_title('Prediction on unseen soils');ax.axhline(0,color='black',lw=.7);plt.tight_layout();plt.savefig(out/'fig_accuracy_comparison.png');plt.close()
q=quart[(quart.target=='Corg')&(quart.model.isin(['V4 plsr_t6','V4 plsr_mixed','V4 combined','V6 adaptive']))]
fig,ax=plt.subplots(figsize=(8,4.5))
for m,g in q.groupby('model'):
 g=g.set_index('quartile').reindex(['Q1 driest','Q2','Q3','Q4 wettest']);ax.plot(g.index,g.mae,marker='o',label=m)
ax.set_ylabel('SOC mean absolute error');ax.set_xlabel('Measured water-content quartile');ax.set_title('Soil organic carbon: errors along the moisture gradient');ax.legend(fontsize=8);plt.tight_layout();plt.savefig(out/'fig_soc_moisture_mae.png');plt.close()
ss=st[(st.validation=='sample_grouped')&(st.target=='Corg')&(st.model.isin(['V4 plsr_mixed','V4 combined','V6 adaptive']))].groupby('model').agg(median_sd=('pred_sd','median'),median_mae=('mae','median'))
ax=ss.plot.scatter(x='median_sd',y='median_mae',figsize=(6,4));
for name,row in ss.iterrows():ax.annotate(name,(row.median_sd,row.median_mae),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.set_xlabel('Median within-soil predicted SOC SD');ax.set_ylabel('Median within-soil SOC MAE');ax.set_title('Stability must be interpreted together with accuracy');plt.tight_layout();plt.savefig(out/'fig_stability_vs_accuracy.png');plt.close()

# compact paper-oriented markdown report
soc=met[(met.validation=='sample_grouped')&(met.target=='Corg')].set_index('model')
site=met[(met.validation=='site_grouped')&(met.target=='Corg')].set_index('model')
def val(m,frame,col):return float(frame.loc[m,col])
lines=['# Soil moisture robustness: evidence synthesis','',
 'Exploratory analysis of previously saved V4/V6 **out-of-fold** predictions; no new model was trained. Physical soil (`sample_id`) is the bootstrap unit for soil-grouped CV, and sampling site (`site_id`) for site-grouped CV. These confidence intervals are **not** independent confirmation after extensive architecture exploration.','',
 f'- {len(left)} spectral observations, {left.sample_id.nunique()} physical soils, {left.site_id.nunique()} sites.',
 f'- V4/V6 sample-group fold labels identical for {fold_agreement:.1%} of observations.',
 '- In each five-fold grouped CV, a soil is absent from the training fold used to predict it. Observation-level spectra are **not** independent replicates.','',
 '## SOC generalization','', '| Model | Soil-grouped R² | Soil-grouped MAE | Site-grouped R² |','|---|---:|---:|---:|']
for m in variants:
 lines.append(f'| {m} | {val(m,soc,"r2"):.3f} | {val(m,soc,"mae"):.3f} | {val(m,site,"r2"):.3f} |')
lines+=['','## Paired physical-soil bootstrap for SOC','',
 'Positive MAE reduction means model B had less error than model A. Percentile intervals resample entire physical soils for soil-grouped CV and entire sites for site-grouped CV (never individual spectra). Results are **exploratory**, and a CI not spanning zero does not correct for choosing models after evaluating them.','',
 '| Validation | Model A → Model B | MAE reduction | Cluster 95% CI |','|---|---|---:|---:|']
for va,x,y,d,lo,hi in records:lines.append(f'| {va} | {x} → {y} | {d:.3f} | [{lo:.3f}, {hi:.3f}] |')
lines+=['','## Critical interpretation','',
 '- Comparing mixed-moisture training against endpoint-only training **confounds moisture diversity and number of spectral training examples**. It supports the practical benefit of the whole calibration strategy, not an isolated causal estimate for moisture augmentation.','- Do not claim complete removal of water interference or universal success across 11 outputs.','- The physical soil is the experimental unit. Moisture quartiles consist of correlated repeat observations, so quartile MAEs are descriptive unless uncertainty is evaluated at the soil level.','- The final all-63-soil checkpoint has no independent test score. Cross-validated scores describe its learning procedure, not direct testing of that single fit.','- Property reference values are reused across a soil drying trajectory; ensure they reasonably remain stable during drying, especially biologically labile parameters.','', '## Files','',
 '- `model_metrics.csv`: all model/property/validation R², MAE, RMSE','- `paired_soil_bootstrap.csv`: paired soil-cluster bootstrap intervals for MAE differences','- `moisture_quartile_errors.csv`: error at measured WC quartiles','- `soil_trajectory_stability.csv`: prediction stability and accuracy per physical soil','- Figures: accuracy, moisture-stratified SOC errors, stability versus accuracy','',
 '## Manuscript-ready conservative interpretation','',
 'The study uses repeated hyperspectral measurements during drying to expose the calibration model to substantial moisture-induced spectral changes. In grouped five-fold cross-validation, mixed-moisture training retained useful SOC accuracy on soils not included in the respective training folds. The endpoint-only PLSR comparator deteriorated sharply at high moisture; however, its smaller training set prevents attributing the entire difference specifically to moisture robustness. Consistency-aware deep models showed differences in the accuracy–stability tradeoff, with benefits depending on the target and validation regime. These exploratory results motivate moisture-diverse calibration but do not establish universal removal of moisture interference or superiority to PLSR.']
(out/'EVIDENCE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[:24]));print('Wrote',out)