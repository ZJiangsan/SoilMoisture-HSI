#!/usr/bin/env python3
"""RF and RBF-SVR soil spectroscopy baselines; nested stratified GROUP CV.
Fits normalization and hyperparameters only on the outer training partition.
Outputs compatible with V4/V6 predictions.csv, with exact same outer CV seed/folds.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGETS=['clay','silt','sand','Corg','Canorg','Ctot','Corg_Clay','DOC','Cmic','FeDi','FeOx']

def load(data):
    labels=pd.read_excel(data/'AustrainSoilChemicalWaterContent_labels.xlsx')
    hsi=pd.read_excel(data/'AustrainSoilHSI_in_differentWaterContent.xlsx')
    if len(labels)!=len(hsi):raise ValueError('Workbook row counts do not match')
    ids=labels['Soil_Code'].astype(str).str.strip().to_numpy()
    if not np.array_equal(ids,hsi['treatments'].astype(str).str.strip().to_numpy()):raise ValueError('Soil rows misaligned')
    water=pd.to_numeric(labels['WC'],errors='coerce').to_numpy(float)
    if not np.allclose(water,pd.to_numeric(hsi['WC'],errors='coerce').to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-5):raise ValueError('Water rows misaligned')
    cols=sorted((c for c in hsi.columns if str(c).startswith('X') and str(c)[1:].isdigit()),key=lambda v:int(str(v)[1:]))
    if len(cols)!=2151:raise ValueError('Expected 2151 wavelength bands')
    x=hsi[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    y=labels[TARGETS].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not np.isfinite(y).all():raise ValueError('This baseline requires finite property labels')
    sites=labels['location'].astype(str).str.strip().to_numpy()
    strata=labels['class'].astype(str).str.strip().to_numpy()
    times=hsi['time'].astype(str).to_numpy()
    return x,y,water,ids,sites,strata,times

def configurations(model,quick):
    if model=='rf':
        c=[{'min_samples_leaf':leaf,'max_features':features,'max_depth':None} for leaf,features in [(1,.3),(2,.6),(3,1.0),(5,.3)]]
    else:
        c=[{'pca':pca,'C':C,'gamma':'scale','epsilon':.1} for pca,C in [(24,1.),(24,10.),(48,1.),(48,10.),(80,1.),(80,10.)]]
    return c[:1] if quick else c

def train_predict(model,config,xfit,yfit,xpred,seed,trees,threads):
    yi=np.mean(yfit,axis=0);ys=np.std(yfit,axis=0);ys=np.where(ys>1e-8,ys,1.)
    yn=(yfit-yi)/ys
    if model=='rf':
        prep=SimpleImputer(strategy='median',add_indicator=False)
        xx=prep.fit_transform(xfit);xp=prep.transform(xpred)
        rf=RandomForestRegressor(n_estimators=trees,min_samples_leaf=config['min_samples_leaf'],
           max_features=config['max_features'],max_depth=config['max_depth'],
           random_state=seed,n_jobs=threads)
        rf.fit(xx,yn)
        p=rf.predict(xp)
    else:
        prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),
           ('pca',PCA(n_components=min(config['pca'],xfit.shape[0]-1,xfit.shape[1]),svd_solver='randomized',random_state=seed))])
        xx=prep.fit_transform(xfit);xp=prep.transform(xpred)
        p=np.empty((len(xpred),yfit.shape[1]))
        for j in range(yfit.shape[1]):
            svr=SVR(kernel='rbf',C=config['C'],epsilon=config['epsilon'],gamma=config['gamma'])
            svr.fit(xx,yn[:,j]);p[:,j]=svr.predict(xp)
    return p*ys+yi

def normalized_score(y,p,refstd):
    return float(np.mean(((y-p)/np.maximum(refstd,1e-8))**2))

def aggregate(pred):
    output=[];stability=[];bins=[];slopes=[]
    for (v,model),df in pred.groupby(['validation','variant']):
        wc_bins=pd.qcut(df.water_content,4,duplicates='drop')
        for target in TARGETS:
            yy=df['true_'+target].to_numpy(float);pp=df['pred_'+target].to_numpy(float)
            good=np.isfinite(yy)&np.isfinite(pp)
            output.append(dict(validation=v,variant=model,target=target,n=int(good.sum()),
                 r2=float(r2_score(yy[good],pp[good])),rmse=float(np.sqrt(mean_squared_error(yy[good],pp[good]))),mae=float(mean_absolute_error(yy[good],pp[good]))))
            for b in wc_bins.cat.categories:
                mask=(wc_bins==b).to_numpy()&good
                bins.append(dict(validation=v,variant=model,target=target,moisture_bin=str(b),n=int(mask.sum()),mae=float(np.mean(np.abs(yy[mask]-pp[mask])))))
            for soil,grp in df.groupby('sample_id'):
                if len(grp)<2:continue
                ysoil=grp['true_'+target].to_numpy(float);psoil=grp['pred_'+target].to_numpy(float)
                stability.append(dict(validation=v,variant=model,target=target,sample_id=soil,n=len(grp),
                    within_sample_prediction_sd=float(np.std(psoil,ddof=1)),sample_bias=float(np.mean(psoil-ysoil)),sample_mae=float(np.mean(np.abs(psoil-ysoil)))))
                wsoil=grp.water_content.to_numpy(float)
                if len(grp)>=3 and np.ptp(wsoil)>1e-8:
                    slopes.append(dict(validation=v,variant=model,sample_id=soil,target=target,n=len(grp),
                         moisture_span=float(np.ptp(wsoil)),prediction_slope_per_wc=float(np.polyfit(wsoil,psoil,1)[0]),mae=float(np.mean(np.abs(psoil-ysoil)))))
    return output,stability,bins,slopes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=Path('soil_moisture_v7_rf_svr_results'))
    ap.add_argument('--models',nargs='+',choices=['rf','svr'],default=['rf','svr'])
    ap.add_argument('--validations',nargs='+',choices=['sample_grouped','site_grouped'],default=['sample_grouped','site_grouped'])
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--inner-folds',type=int,default=3)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--tuning-trees',type=int,default=100)
    ap.add_argument('--final-trees',type=int,default=300)
    ap.add_argument('--smoke',action='store_true',help='Only first outer fold, first candidate. Not manuscript results.')
    ap.add_argument('--compare',type=Path,default=None,help='Optional V4 predictions.csv to check identical held-out folds')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    x,y,w,ids,sites,strata,times=load(args.data_dir)
    print(f'Data: {len(x)} spectra, {len(np.unique(ids))} soils, {len(np.unique(sites))} sites, {x.shape[1]} bands',flush=True)
    old=pd.read_csv(args.compare) if args.compare is not None else None
    predictions=[]; fold_results=[];cv_results=[]; audits=[]
    for validation in args.validations:
        group=ids if validation=='sample_grouped' else sites
        outer=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
        for fold,(itr,ite) in enumerate(outer.split(x,strata,groups=group),1):
            if args.smoke and fold>1:break
            if set(ids[itr])&set(ids[ite]):raise RuntimeError('Soil leakage')
            if validation=='site_grouped' and set(sites[itr])&set(sites[ite]):raise RuntimeError('Site leakage')
            if old is not None:
                previous=old[(old.validation==validation)&(old.fold==fold)]
                if set(previous.row_index.astype(int))!=set(ite):raise RuntimeError(f'Fold mismatch with V4: {validation} fold {fold}')
            for part,rows in [('train',itr),('test',ite)]:
                for category in np.unique(strata):
                    loc=rows[strata[rows]==category]
                    audits.append(dict(validation=validation,fold=fold,partition=part,category=category,
                        observations=len(loc),unique_soils=len(np.unique(ids[loc])),unique_sites=len(np.unique(sites[loc]))))
            print(f'{validation} fold={fold}: train {len(itr)} test {len(ite)}',flush=True)
            inner_n=min(args.inner_folds,len(np.unique(group[itr])))
            inner=StratifiedGroupKFold(n_splits=inner_n,shuffle=True,random_state=args.seed+fold)
            inner_pairs=list(inner.split(x[itr],strata[itr],groups=group[itr]))
            for model in args.models:
                best=float('inf');bestcfg=None
                for ci,cfg in enumerate(configurations(model,args.smoke)):
                    scores=[]
                    for j,(a,b) in enumerate(inner_pairs):
                        fit,valid=itr[a],itr[b]
                        prediction=train_predict(model,cfg,x[fit],y[fit],x[valid],args.seed+fold*100+j,
                              args.tuning_trees,args.threads)
                        score=normalized_score(y[valid],prediction,y[fit].std(axis=0))
                        scores.append(score)
                    score=float(np.mean(scores))
                    cv_results.append(dict(validation=validation,fold=fold,variant=model,configuration=json.dumps(cfg),mean_normalized_mse=score,inner_fold_mses=json.dumps(scores)))
                    print(f'  {model} cfg {ci+1}: {score:.5f} {cfg}',flush=True)
                    if score<best:best=score;bestcfg=cfg
                p=train_predict(model,bestcfg,x[itr],y[itr],x[ite],args.seed+fold*1000,
                         args.final_trees,args.threads)
                row=pd.DataFrame({'row_index':ite,'sample_id':ids[ite],'site_id':sites[ite],
                       'water_content':w[ite],'time':times[ite],'validation':validation,'fold':fold,
                       'variant':model,'selected_epochs':np.nan,'plsr_components':np.nan})
                for j,t in enumerate(TARGETS):row['true_'+t]=y[ite,j];row['pred_'+t]=p[:,j]
                predictions.append(row)
                for j,t in enumerate(TARGETS):
                    fold_results.append(dict(validation=validation,fold=fold,variant=model,target=t,n=len(ite),
                          r2=float(r2_score(y[ite,j],p[:,j])),rmse=float(np.sqrt(mean_squared_error(y[ite,j],p[:,j]))),
                          mae=float(mean_absolute_error(y[ite,j],p[:,j]))))
                print('  selected',model,bestcfg,'inner normalized MSE',best,flush=True)
    allpred=pd.concat(predictions,ignore_index=True).sort_values(['validation','variant','row_index'])
    allpred.to_csv(args.output/'predictions.csv',index=False)
    metrics,stability,bins,slopes=aggregate(allpred)
    for name,records in [('overall_metrics',metrics),('fold_metrics',fold_results),('inner_cv_results',cv_results),
                         ('stratification_balance',audits),('within_sample_stability',stability),
                         ('moisture_stratified_metrics',bins),('within_soil_moisture_slopes',slopes)]:
        pd.DataFrame(records).to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'configuration.json').write_text(json.dumps({**vars(args),'output':str(args.output),'data_dir':str(args.data_dir),
        'compare':str(args.compare),'targets':TARGETS},indent=2,default=str))
    print(pd.DataFrame(metrics).query("target=='Corg'").to_string(index=False),flush=True)
    print('Saved to',args.output,flush=True)
if __name__=='__main__':main()