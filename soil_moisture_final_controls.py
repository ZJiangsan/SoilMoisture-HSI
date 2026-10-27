#!/usr/bin/env python3
"""Computational moisture-control experiment on existing 2023 soil spectra.

No new physical measurements. Three calibration regimes per outer fold:
  endpoint: one t6/driest observation per training soil
  random_one: one randomly chosen observation per training soil (balanced N)
  all_moisture: all observations for every training soil (diversity + N)
All predict the SAME spectra of completely held-out soils.
Repeated random_one selections quantify sensitivity to the chosen moisture observation.
Optional: paired cluster bootstrap comparison of EXISTING saved model predictions.

Dependencies: numpy pandas scipy scikit-learn openpyxl matplotlib.
"""
import argparse, json, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

TARGETS=['clay','silt','sand','Corg','Canorg','Ctot','Corg_Clay','DOC','Cmic','FeDi','FeOx']

def read_data(root):
    lab=pd.read_excel(root/'AustrainSoilChemicalWaterContent_labels.xlsx')
    hsi=pd.read_excel(root/'AustrainSoilHSI_in_differentWaterContent.xlsx')
    assert len(lab)==len(hsi)
    ids=lab.Soil_Code.astype(str).str.strip().to_numpy()
    assert np.array_equal(ids,hsi.treatments.astype(str).str.strip().to_numpy())
    w=pd.to_numeric(hsi.WC,errors='coerce').to_numpy(float)
    assert np.allclose(w,pd.to_numeric(lab.WC).to_numpy(float),atol=1e-5)
    cols=sorted([c for c in hsi.columns if str(c).startswith('X') and str(c)[1:].isdigit()],key=lambda a:int(a[1:]))
    X=hsi[cols].to_numpy(float)
    Y=lab[TARGETS].to_numpy(float)
    if len(cols)!=2151 or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Unexpected number of bands, missing or nonfinite values; inspect data.')
    return X,Y,w,ids,lab['location'].astype(str).to_numpy(),lab['class'].astype(str).to_numpy(),hsi.time.astype(str).to_numpy()

def outer_folds(yclass,ids,seed,nfolds):
    sg=StratifiedGroupKFold(n_splits=nfolds,shuffle=True,random_state=seed)
    yield from sg.split(np.zeros(len(ids)),yclass,ids)

def one_per_soil(train,ids,w,time,mode,rng):
    chosen=[]
    for s in np.unique(ids[train]):
        opts=train[ids[train]==s]
        if mode=='endpoint':
            dry=opts[time[opts]=='t6']
            # Prefer explicitly declared endpoint; if unavailable, use driest measured observation.
            chosen.append(int(dry[np.argmin(w[dry])] if len(dry) else opts[np.argmin(w[opts])]))
        elif mode=='random_one': chosen.append(int(rng.choice(opts)))
        else: raise ValueError(mode)
    return np.array(sorted(chosen),dtype=int)

def fit_plsr(X,Y,train,components):
    scaler=StandardScaler().fit(X[train]); xs=scaler.transform(X[train]);
    ymu=Y[train].mean(0); ysd=Y[train].std(0);ysd[ysd<1e-12]=1.
    n=min(int(components),len(train)-1,X.shape[1]);m=PLSRegression(n_components=n,scale=False)
    m.fit(xs,(Y[train]-ymu)/ysd)
    return lambda idx:m.predict(scaler.transform(X[idx]))*ysd+ymu

def fit_svr(X,Y,train,c,pca,epsilon):
    # All scalers and PCA fit on training data only.
    n_components=min(pca,len(train)-1,X.shape[1]);
    # If very small training sample, PCA still valid if >= 2.
    model=Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=n_components,svd_solver='randomized',random_state=12)),('reg',MultiOutputRegressor(SVR(kernel='rbf',C=c,gamma='scale',epsilon=epsilon)))])
    ym=Y[train].mean(0);ys=Y[train].std(0);ys[ys<1e-12]=1.
    model.fit(X[train],(Y[train]-ym)/ys)
    return lambda idx:model.predict(X[idx])*ys+ym

def tune(X,Y,ids,w,time,yclass,train,method,mode,rng_seed):
    # Grouped inner CV, same calibration regime applied inside each split.
    n_groups=len(np.unique(ids[train]))
    if n_groups<8: raise ValueError('Too few training groups')
    inn=GroupKFold(n_splits=3)
    candidates=([('plsr',c) for c in (2,5,10)] if method=='plsr'
                else [('svr',(c,p)) for c in (0.3,3.,30.) for p in (8,20)])
    # A predetermined subset of targets defines tuning score; equal normalized errors per target.
    sd=np.std(Y[train],axis=0);sd[sd<1e-12]=1.
    best=(np.inf,None)
    for model_type,param in candidates:
        losses=[]
        for split_i,(tr_loc,va_loc) in enumerate(inn.split(np.zeros(len(train)),groups=ids[train])):
            tr=train[tr_loc]; va=train[va_loc]
            if mode!='all_moisture':
                tr=one_per_soil(tr,ids,w,time,mode,np.random.default_rng(rng_seed+split_i+101))
            try:
                pred=(fit_plsr(X,Y,tr,param) if model_type=='plsr'
                      else fit_svr(X,Y,tr,param[0],param[1],0.1))(va)
                losses.append(float(np.mean(np.square((pred-Y[va])/sd))))
            except (ValueError,ArithmeticError) as exc:
                warnings.warn(str(exc));losses.append(np.inf)
        sc=np.mean(losses)
        if sc<best[0]: best=(sc,param)
    if best[1] is None: raise RuntimeError('All hyperparameter candidates failed')
    return best

def bootstrap_summary(predictions,out,bootstrap,seed):
    if predictions is None:return
    # Saved results from prior V4, V7 and V8 experiments, paired by row_index.
    frames=[]
    for label,path,variants in [
        ('V4',predictions[0],['plsr_mixed','plsr_t6','combined']),
        ('V7',predictions[1],['rf','svr']),
        ('V8',predictions[2],['cnn1d','transformer'])]:
        if not path.exists():
            print('Skipping unavailable archive:',path);continue
        with zipfile.ZipFile(path) as z:
            items=[n for n in z.namelist() if n.endswith('predictions.csv')]
            if not items: raise ValueError(f'{path} has no predictions.csv')
            d=pd.read_csv(z.open(items[0]))
        d=d[(d.validation=='sample_grouped')&d.variant.isin(variants)].copy()
        d['model']=label+'_'+d.variant
        frames.append(d)
    if not frames:return
    d=pd.concat(frames,ignore_index=True)
    results=[]
    for model,g in d.groupby('model'):
        if g.row_index.nunique()!=len(g):raise ValueError(f'Duplicate predictions: {model}')
        for name in TARGETS:
            y=g['true_'+name].to_numpy();p=g['pred_'+name].to_numpy()
            results.append(dict(model=model,target=name,r2=r2_score(y,p),mae=np.mean(np.abs(y-p)),rmse=np.sqrt(np.mean((y-p)**2)),n=len(y)))
    pd.DataFrame(results).to_csv(out/'existing_model_metrics.csv',index=False)
    # Soil resampling used for uncertainty on paired differences in absolute error.
    rng=np.random.default_rng(seed);groups={m:g.set_index('row_index').sort_index() for m,g in d.groupby('model')}
    pairs=[('V8_cnn1d',x) for x in ('V4_plsr_mixed','V7_svr','V7_rf','V8_transformer','V4_combined')]
    rows=[]
    for first,second in pairs:
        if first not in groups or second not in groups:continue
        a,b=groups[first].align(groups[second],join='inner',axis=0)
        if len(a)!=426: raise ValueError(f'Expected 426 paired out-of-fold rows, got {len(a)}')
        assert np.array_equal(a.sample_id.to_numpy(),b.sample_id.to_numpy())
        soils=a.sample_id.unique();soil_idx=[np.where(a.sample_id.to_numpy()==s)[0] for s in soils]
        draw=rng.integers(len(soils),size=(bootstrap,len(soils)))
        counts=np.array([len(i) for i in soil_idx])
        for name in TARGETS:
            ea=np.abs(a['true_'+name].to_numpy()-a['pred_'+name].to_numpy())
            eb=np.abs(b['true_'+name].to_numpy()-b['pred_'+name].to_numpy())
            delta=np.array([(eb[i]-ea[i]).sum() for i in soil_idx])
            effect=delta.sum()/counts.sum()  # positive means CNN better
            samples=delta[draw].sum(axis=1)/counts[draw].sum(axis=1)
            lo,hi=np.quantile(samples,[.025,.975])
            rows.append(dict(target=name,model_a=first,model_b=second,mae_reduction_a_vs_b=effect,ci_low=lo,ci_high=hi,n_soils=len(soils),exploratory=True))
    pd.DataFrame(rows).to_csv(out/'paired_model_uncertainty.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('soil_moisture_final_controls'))
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--random-repeats',type=int,default=3)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--models',nargs='+',choices=['plsr','svr'],default=['plsr','svr'])
    parser.add_argument('--existing-v4',type=Path,default=None)
    parser.add_argument('--existing-v7',type=Path,default=None)
    parser.add_argument('--existing-v8',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=10)
    args=parser.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    X,Y,w,ids,sites,yclass,time=read_data(args.data_dir)
    uniq=np.unique(ids)
    assert len(X)==426 and len(uniq)==63, f'Unexpected 2023 dataset: n={len(X)}, independent soils={len(uniq)}'
    pd.DataFrame({'soil':ids,'wc':w,'time':time}).groupby('soil').agg(n=('wc','size'),wc_min=('wc','min'),wc_max=('wc','max')).to_csv(out/'measured_moisture_coverage.csv')
    records=[];tuning=[]
    folds=list(outer_folds(yclass,ids,args.seed,args.folds))
    for fold,(tr,te) in enumerate(folds,1):
        assert not set(ids[tr])&set(ids[te]);assert len(te)>0
        for mode in ('endpoint','random_one','all_moisture'):
            repeats=args.random_repeats if mode=='random_one' else 1
            for rep in range(repeats):
                rng=np.random.default_rng(args.seed+fold*1000+rep)
                fitidx=(tr if mode=='all_moisture' else one_per_soil(tr,ids,w,time,mode,rng))
                print(f'fold={fold} regime={mode} rep={rep} ntrain={len(fitidx)} ntest={len(te)}',flush=True)
                for method in args.models:
                    # Tune hyperparameters entirely inside outer training pool.
                    score,param=tune(X,Y,ids,w,time,yclass,tr,method,mode,args.seed+fold*1000+rep)
                    fitted=(fit_plsr(X,Y,fitidx,param) if method=='plsr' else fit_svr(X,Y,fitidx,param[0],param[1],0.1))
                    p=fitted(te)
                    tuning.append(dict(fold=fold,regime=mode,rep=rep,model=method,best_param=str(param),inner_normalized_mse=score,ntrain=len(fitidx),n_soils_train=len(set(ids[fitidx]))))
                    for row,j in enumerate(te):
                        rec=dict(row_index=int(j),sample_id=ids[j],site_id=sites[j],water_content=w[j],time=time[j],fold=fold,regime=mode,rep=rep,model=method)
                        for k,name in enumerate(TARGETS):rec['true_'+name]=Y[j,k];rec['pred_'+name]=p[row,k]
                        records.append(rec)
    raw=pd.DataFrame(records);raw.to_csv(out/'controlled_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'hyperparameter_selection.csv',index=False)
    metrics=[]
    for (regime,rep,model),g in raw.groupby(['regime','rep','model']):
        for name in TARGETS:
            y=g['true_'+name].to_numpy(); p=g['pred_'+name].to_numpy()
            metrics.append(dict(regime=regime,rep=rep,model=model,target=name,n=len(g),r2=r2_score(y,p),mae=np.mean(np.abs(y-p)),rmse=np.sqrt(np.mean((y-p)**2))))
    pd.DataFrame(metrics).to_csv(out/'controlled_metrics.csv',index=False)
    # Paired comparison with equal number of training spectra: random_one vs endpoint.
    bs=[];rng=np.random.default_rng(args.seed)
    for model in args.models:
        a=raw[(raw.model==model)&(raw.regime=='endpoint')].set_index('row_index').sort_index()
        if len(a)!=426:raise ValueError('Missing endpoint OOF predictions')
        for rep in range(args.random_repeats):
            b=raw[(raw.model==model)&(raw.regime=='random_one')&(raw.rep==rep)].set_index('row_index').sort_index()
            assert np.array_equal(a.index,b.index)
            idx=[np.where(a.sample_id.to_numpy()==s)[0] for s in a.sample_id.unique()]
            counts=np.array([len(j) for j in idx]); draw=rng.integers(len(idx),size=(args.bootstrap,len(idx)))
            for name in TARGETS:
                ea=np.abs(a['true_'+name]-a['pred_'+name]).to_numpy()
                eb=np.abs(b['true_'+name]-b['pred_'+name]).to_numpy()
                delta=np.array([(ea[j]-eb[j]).sum() for j in idx])
                boot=delta[draw].sum(axis=1)/counts[draw].sum(axis=1)
                low,high=np.quantile(boot,[.025,.975])
                bs.append(dict(model=model,rep=rep,target=name,endpoint_mae=ea.mean(),random_one_mae=eb.mean(),mae_reduction_random_vs_endpoint=delta.sum()/counts.sum(),ci_low=low,ci_high=high,clusters=len(idx)))
    pd.DataFrame(bs).to_csv(out/'balanced_moisture_effect_bootstrap.csv',index=False)
    # Moisture quartile plots, same measured moisture boundaries for all groups.
    edges=np.quantile(w,[0,.25,.5,.75,1]);edges=np.unique(edges)
    raw['moisture_q']=pd.cut(raw.water_content,bins=edges,include_lowest=True,labels=[f'Q{i+1}' for i in range(len(edges)-1)])
    qm=[]
    for (reg,rep,mod,q),g in raw.groupby(['regime','rep','model','moisture_q'],observed=True):
        for name in TARGETS:qm.append(dict(regime=reg,rep=rep,model=mod,quartile=str(q),target=name,n=len(g),mae=np.mean(np.abs(g['true_'+name]-g['pred_'+name]))))
    qdf=pd.DataFrame(qm);qdf.to_csv(out/'controlled_moisture_quartiles.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for regime,g in qdf[(qdf.model=='plsr')&(qdf.target=='Corg')].groupby('regime'):
        tab=g.groupby('quartile').mae.mean().reindex([f'Q{i+1}' for i in range(len(edges)-1)])
        ax.plot(tab.index,tab.values,marker='o',label=regime)
    ax.set_ylabel('SOC MAE');ax.set_xlabel('Measured WC quartile');ax.set_title('Controlled moisture calibration; unseen soils');ax.legend();fig.tight_layout();fig.savefig(out/'soc_moisture_control.png',dpi=180);plt.close(fig)
    if all(x is not None for x in (args.existing_v4,args.existing_v7,args.existing_v8)):
        bootstrap_summary((args.existing_v4,args.existing_v7,args.existing_v8),out,args.bootstrap,args.seed)
    (out/'interpretation_notes.txt').write_text(
       'endpoint vs random_one: matched number of physical soils AND training spectra; compares choice of moisture condition, not benefit of seven spectra per soil.\n'
       'all_moisture vs single-spectrum conditions: mixes moisture diversity with more observations, so do NOT ascribe all gains to moisture diversity alone.\n'
       'All metrics are out-of-fold predictions for unseen physical soils. Inner tuning never uses outer test soils.\n'
       'Confidence intervals cluster-bootstrap soils and are exploratory after extensive model searching.\n'
       't6 is historical drying endpoint; verify it is indeed the driest; fallback is minimum measured WC.\n'
       'Only computational reanalysis of 2023 measurements; no new soil experiments.\n')
    print('Finished:',out.resolve())

if __name__=='__main__':main()