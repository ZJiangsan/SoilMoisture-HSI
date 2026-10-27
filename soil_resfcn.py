#!/usr/bin/env python3
"""Leakage-safe multi-target soil ResFCN and moisture-invariance study.

Requirements: torch, numpy, pandas, openpyxl, scikit-learn, matplotlib.
All repeated spectra from the same physical soil stay in the same CV partition.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL='AustrainSoilChemicalWaterContent_labels.xlsx'
SPECTRA='AustrainSoilHSI_in_differentWaterContent.xlsx'
TARGETS=['clay','silt','sand','Corg','Canorg','Ctot','Corg_Clay','DOC','Cmic','FeDi','FeOx']
VARIANTS=['plain','aux','consistency','combined']
# Baselines are separate from the neural network variants.
BASELINES=['plsr_mixed','plsr_t6']


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.fn=nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.GELU(),
                              nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(hidden)
    def forward(self,x):
        return self.norm(x+self.fn(x))


class SoilResFCN(nn.Module):
    """New compact residual fully-connected multi-output model (not exact historical reproduction)."""
    def __init__(self, bands, outputs, hidden=256, blocks=2, dropout=.15):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(bands,hidden),nn.LayerNorm(hidden),nn.GELU(),
                                   nn.Dropout(dropout),*[ResidualMLPBlock(hidden,dropout) for _ in range(blocks)])
        self.props=nn.ModuleList([nn.Sequential(nn.Linear(hidden,64), nn.GELU(), nn.Linear(64,1)) for _ in range(outputs)])
        self.moisture=nn.Linear(hidden,1)
    def forward(self,x):
        h=self.encoder(x)
        return torch.cat([head(h) for head in self.props],dim=1),self.moisture(h).squeeze(-1)


def load_data(data_dir):
    lab=pd.read_excel(data_dir/LABEL)
    hsi=pd.read_excel(data_dir/SPECTRA)
    if len(lab)!=len(hsi): raise ValueError('Label and spectral row counts differ')
    ids=lab['Soil_Code'].astype(str).str.strip().to_numpy()
    ids_h=hsi['treatments'].astype(str).str.strip().to_numpy()
    water=pd.to_numeric(lab['WC'],errors='coerce').to_numpy(float)
    water_h=pd.to_numeric(hsi['WC'],errors='coerce').to_numpy(float)
    if not np.array_equal(ids,ids_h) or not np.allclose(water,water_h,equal_nan=True,rtol=1e-6,atol=1e-5):
        raise ValueError('Soil IDs or WC disagree across workbooks: refusing row-wise matching')
    spec_cols=sorted([c for c in hsi if str(c).startswith('X') and str(c)[1:].isdigit()],key=lambda c:int(str(c)[1:]))
    if len(spec_cols)!=2151: raise ValueError(f'Expected 2151 spectral bands, found {len(spec_cols)}')
    X=hsi[spec_cols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    Y=lab[TARGETS].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    if not np.isfinite(water).all(): raise ValueError('Missing moisture labels')
    sites=lab['location'].astype(str).str.strip().to_numpy()
    # Essential for consistency loss: a property's ground truth must remain fixed across drying.
    for j,name in enumerate(TARGETS):
        for soil in np.unique(ids):
            a=Y[ids==soil,j]; a=a[np.isfinite(a)]
            if len(a)>1 and np.ptp(a)>max(1e-5,1e-5*abs(np.median(a))):
                raise ValueError(f'Target {name} changes within physical sample {soil}; inspect first')
    soil_class=lab["class"].astype(str).str.strip().to_numpy()
    management=lab["soil_type"].astype(str).str.strip().to_numpy()
    return X,Y,water,ids,sites,hsi["time"].astype(str).to_numpy(),soil_class,management


def preprocess_fit(X,Y,W,training):
    a=X[training]; finite=np.isfinite(a)
    med=np.nanmedian(np.where(finite,a,np.nan),axis=0)
    med=np.where(np.isfinite(med),med,0).astype(np.float32)
    clean=np.where(finite,a,med)
    xm=clean.mean(0,dtype=np.float64).astype(np.float32)
    xs=clean.std(0,dtype=np.float64).astype(np.float32)
    xs=np.where(xs>1e-7,xs,1).astype(np.float32)
    ym=np.nanmean(Y[training],axis=0)
    ys=np.nanstd(Y[training],axis=0)
    ym=np.where(np.isfinite(ym),ym,0).astype(np.float32)
    ys=np.where(np.isfinite(ys)&(ys>1e-8),ys,1).astype(np.float32)
    wm=float(np.mean(W[training])); ws=max(float(np.std(W[training])),1e-8)
    def transform(rows):
        raw=X[rows]; raw=np.where(np.isfinite(raw),raw,med)
        return (np.clip((raw-xm)/xs,-10,10).astype(np.float32),
                ((Y[rows]-ym)/ys).astype(np.float32),
                ((W[rows]-wm)/ws).astype(np.float32))
    return transform,(ym,ys),{'band_median':med,'band_mean':xm,'band_std':xs,'label_mean':ym,'label_std':ys,'wc_mean':wm,'wc_std':ws}


def make_batches(indices,ids,number_soils,rng):
    groups=np.unique(ids[indices]).copy(); rng.shuffle(groups)
    for offset in range(0,len(groups),number_soils):
        current=groups[offset:offset+number_soils]
        # Include ALL moisture levels of every selected physical soil in one batch.
        yield indices[np.isin(ids[indices],current)]


def loss_fn(model,x,y,w,group_ids,variant,aux_weight,cons_weight):
    p,m=model(x)
    valid=torch.isfinite(y)
    diff=torch.where(valid,p-torch.nan_to_num(y,nan=0.),torch.zeros_like(p))
    denom=valid.sum(0).clamp(min=1)
    active=valid.any(0)
    supervised=((diff.square().sum(0)/denom)*active).sum()/active.sum().clamp(min=1)
    loss=supervised
    if variant in ('aux','combined'):
        loss=loss+aux_weight*torch.mean((m-w).square())
    if variant in ('consistency','combined'):
        # Exact same normalized property output should hold for changing moisture in one soil.
        terms=[]
        for soil in np.unique(group_ids):
            mask=torch.as_tensor(group_ids==soil,device=p.device)
            if mask.sum()>1:
                pred=p[mask]
                v=valid[mask]
                cnt=v.sum(0).clamp(min=1)
                mean=(pred*v).sum(0)/cnt
                within=(((pred-mean).square())*v).sum(0)/cnt
                active=v.any(0)
                if active.any(): terms.append(within[active].mean())
        if terms: loss=loss+cons_weight*torch.stack(terms).mean()
    return loss,supervised


def train(X,Y,W,ids,fit_idx,variant,args,device,epochs,watch=None,seed=42,lr_replay=None):
    seed_all(seed)
    transform,target_scale,stats=preprocess_fit(X,Y,W,fit_idx)
    # transform entire dataset using TRAINING-only normalization
    xt,yt,wt=transform(np.arange(len(X)))
    tx=torch.from_numpy(xt).to(device)
    ty=torch.from_numpy(yt).to(device)
    tw=torch.from_numpy(wt).to(device)
    model=SoilResFCN(X.shape[1],Y.shape[1],args.hidden,args.blocks,args.dropout).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,mode='min',factor=args.lr_factor,patience=args.lr_patience,
        threshold=args.lr_threshold,threshold_mode='rel',min_lr=args.min_lr
    ) if watch is not None else None)
    best=np.inf;best_ep=epochs;bad=0;rng=np.random.default_rng(seed)
    history=[]
    for epoch in range(1,epochs+1):
        if lr_replay is not None:
            if epoch>len(lr_replay): raise ValueError('LR replay is shorter than requested refit')
            for pg in optim.param_groups: pg['lr']=float(lr_replay[epoch-1])
        lr_used=float(optim.param_groups[0]['lr'])
        model.train(); losses=[]
        for ii in make_batches(fit_idx,ids,args.soils_per_batch,rng):
            optim.zero_grad(set_to_none=True)
            loss,_=loss_fn(model,tx[ii],ty[ii],tw[ii],ids[ii],variant,args.aux_weight,args.cons_weight)
            if not torch.isfinite(loss): raise RuntimeError(f'Invalid training loss at epoch {epoch}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optim.step();losses.append(float(loss.detach().cpu()))
        rec={'epoch':epoch,'learning_rate':lr_used,'training_loss':float(np.mean(losses))}
        if watch is not None:
            model.eval()
            with torch.no_grad():
                pred,_=model(tx[watch]);gold=ty[watch]
                valid=torch.isfinite(gold)
                mse=((pred-torch.nan_to_num(gold,nan=0.)).square()*valid).sum(0)/valid.sum(0).clamp(min=1)
                active=valid.any(0)
                val=float(mse[active].mean().cpu())
            rec['inner_validation_loss']=val
            if val<best-args.min_delta: best=val;best_ep=epoch;bad=0
            else: bad+=1
            scheduler.step(val)
            rec['next_learning_rate']=float(optim.param_groups[0]['lr'])
        history.append(rec)
        if watch is not None and bad>=args.patience: break
    return model,transform,target_scale,history,best_ep


def predict(model,transform,scale,indices,device):
    x,_,_=transform(indices)
    model.eval()
    with torch.no_grad():
        p,m=model(torch.as_tensor(x,device=device))
    return p.cpu().numpy()*scale[1]+scale[0],m.cpu().numpy()


def metrics(y,p):
    good=np.isfinite(y)&np.isfinite(p)
    if sum(good)<3: return {'n':int(sum(good)),'r2':np.nan,'rmse':np.nan,'mae':np.nan}
    a=y[good];b=p[good]
    return {'n':int(len(a)),'r2':float(r2_score(a,b)),
            'rmse':float(np.sqrt(mean_squared_error(a,b))),'mae':float(mean_absolute_error(a,b))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--validations',nargs='+',choices=['sample_grouped','site_grouped'],default=['sample_grouped','site_grouped'])
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--stratify-by',choices=['class','soil_type','none'],default='class',
                    help='Original paper used texture class. Stratify groups on this variable; never on time or WC.')
    ap.add_argument('--max-epochs',type=int,default=600)
    ap.add_argument('--patience',type=int,default=80)
    ap.add_argument('--min-delta',type=float,default=0.0001)
    ap.add_argument('--lr-patience',type=int,default=15)
    ap.add_argument('--lr-factor',type=float,default=0.5)
    ap.add_argument('--min-lr',type=float,default=1e-7)
    ap.add_argument('--lr-threshold',type=float,default=1e-4)
    ap.add_argument('--hidden',type=int,default=192)
    ap.add_argument('--blocks',type=int,default=2)
    ap.add_argument('--dropout',type=float,default=.15)
    ap.add_argument('--lr',type=float,default=.0004)
    ap.add_argument('--weight-decay',type=float,default=.001)
    ap.add_argument('--aux-weight',type=float,default=.2)
    ap.add_argument('--cons-weight',type=float,default=.4)
    ap.add_argument('--soils-per-batch',type=int,default=12)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--baselines',action='store_true',default=True)
    ap.add_argument('--no-baselines',dest='baselines',action='store_false')
    ap.add_argument('--diagnostic-random',action='store_true',help='Observation-random split; not independent-soil validation')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.max_epochs<2 or args.folds<2: raise ValueError('epochs and folds must be >= 2')
    if not (0<args.lr_factor<1) or args.lr_patience<1 or args.patience<=args.lr_patience:
        raise ValueError('Need 0 < lr-factor < 1, lr-patience >= 1 and early stop patience > lr-patience')
    seed_all(args.seed)
    device=torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available')
    out=args.output or args.data_dir/'soil_resfcn_results'
    out.mkdir(parents=True,exist_ok=True)
    (out/'configuration.json').write_text(json.dumps({**vars(args),'data_dir':str(args.data_dir),'output':str(out),'device_used':str(device),'targets':TARGETS},indent=2,default=str))
    print('Loading verified paired soil workbooks ...',flush=True)
    X,Y,W,ids,sites,times,soil_class,management=load_data(args.data_dir)
    strata = soil_class if args.stratify_by=='class' else (management if args.stratify_by=='soil_type' else None)
    split_audit=[]
    print(f'{len(X)} observations; {len(np.unique(ids))} soils; {len(np.unique(sites))} sites; device={device}',flush=True)
    result_rows=[];history_rows=[];pred_frames=[]
    for validation in (args.validations+(['observation_random'] if args.diagnostic_random else [])):
        groups=ids if validation=='sample_grouped' else sites
        if validation=='observation_random':
            # Diagnostic only: allows physical soils to appear on both sides.
            splitter=(StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
                      if strata is not None else KFold(n_splits=args.folds,shuffle=True,random_state=args.seed))
            outer_splits=splitter.split(X,strata) if strata is not None else splitter.split(X)
        elif strata is not None:
            splitter=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
            outer_splits=splitter.split(X,strata,groups=groups)
        else:
            outer_splits=GroupKFold(n_splits=args.folds).split(X,groups=groups)
        for fold,(outer_train,test) in enumerate(outer_splits,start=1):
            if validation!='observation_random': assert not set(ids[outer_train]).intersection(ids[test])
            if validation=='site_grouped': assert not set(sites[outer_train]).intersection(sites[test])
            # Inner tuning split is group-disjoint, with stratification where feasible.
            if validation=='observation_random':
                ai,bi=train_test_split(np.arange(len(outer_train)),test_size=.2,random_state=args.seed+fold,
                    stratify=strata[outer_train] if strata is not None else None)
            elif strata is not None:
                inner_cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed+fold)
                ai,bi=next(inner_cv.split(X[outer_train],strata[outer_train],groups=groups[outer_train]))
            else:
                split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed+fold)
                ai,bi=next(split.split(X[outer_train],groups=groups[outer_train]))
            inner_train=outer_train[ai];inner_val=outer_train[bi]
            if validation!='observation_random': assert not set(ids[inner_train]).intersection(ids[inner_val])
            if validation=='site_grouped': assert not set(sites[inner_train]).intersection(sites[inner_val])
            for partition,rows in [('train',outer_train),('test',test),('inner_train',inner_train),('inner_validation',inner_val)]:
                for cls,count in zip(*np.unique(strata[rows] if strata is not None else soil_class[rows],return_counts=True)):
                    split_audit.append(dict(validation=validation,fold=fold,partition=partition,
                        stratification=args.stratify_by,category=str(cls),observations=int(count),
                        unique_soils=int(np.unique(ids[rows][(strata[rows] if strata is not None else soil_class[rows])==cls]).size),
                        unique_sites=int(np.unique(sites[rows][(strata[rows] if strata is not None else soil_class[rows])==cls]).size)))
            print(f'FOLD {validation} {fold}: train={len(outer_train)} ({len(np.unique(ids[outer_train]))} soils), '
                  f'test={len(test)} ({len(np.unique(ids[test]))} soils)',flush=True)
            if args.baselines:
                for bname in BASELINES:
                    fit_rows=outer_train if bname=='plsr_mixed' else outer_train[times[outer_train]=='t6']
                    if len(fit_rows)<12:
                        print('Skipping insufficient dry observations',bname,len(fit_rows),flush=True);continue
                    # Tune only on the outer training partition, with soil-grouped CV.
                    candidate=[2,5,10,15]
                    ins=(StratifiedGroupKFold(n_splits=min(3,len(np.unique(ids[fit_rows]))),shuffle=True,random_state=args.seed+fold)
                         if strata is not None else GroupKFold(n_splits=min(3,len(np.unique(ids[fit_rows])))))
                    bestscore=float('inf');bestn=None
                    for ncomp in candidate:
                        if ncomp>=len(fit_rows)*.7: continue
                        scores=[]
                        for it,iv in ins.split(X[fit_rows],strata[fit_rows] if strata is not None else None,groups=ids[fit_rows]):
                            if ncomp>=len(it):continue
                            for j in range(len(TARGETS)):
                                good_tr=np.isfinite(Y[fit_rows[it],j]);good_va=np.isfinite(Y[fit_rows[iv],j])
                                if good_tr.sum()<ncomp+2 or not good_va.any():continue
                                pipe=make_pipeline(StandardScaler(),PLSRegression(n_components=ncomp,scale=False))
                                pipe.fit(X[fit_rows[it]][good_tr],Y[fit_rows[it],j][good_tr]);pr=pipe.predict(X[fit_rows[iv]][good_va]).ravel()
                                scale=np.nanstd(Y[fit_rows[it],j]);scores.append(np.mean(((pr-Y[fit_rows[iv],j][good_va])/max(scale,1e-8))**2))
                        if scores and np.mean(scores)<bestscore:bestscore=float(np.mean(scores));bestn=ncomp
                    if bestn is None: raise RuntimeError('No valid PLSR components')
                    pred=np.full((len(test),len(TARGETS)),np.nan)
                    for j in range(len(TARGETS)):
                        good=np.isfinite(Y[fit_rows,j]);pipe=make_pipeline(StandardScaler(),PLSRegression(n_components=bestn,scale=False))
                        pipe.fit(X[fit_rows][good],Y[fit_rows,j][good]);pred[:,j]=pipe.predict(X[test]).ravel()
                    frame=pd.DataFrame({'row_index':test,'sample_id':ids[test],'site_id':sites[test],'water_content':W[test],
                                       'time':times[test],'validation':validation,'fold':fold,'variant':bname,'selected_epochs':np.nan,'plsr_components':bestn})
                    for j,t in enumerate(TARGETS):
                        frame[f'true_{t}']=Y[test,j];frame[f'pred_{t}']=pred[:,j]
                        result_rows.append({'target':t,'validation':validation,'variant':bname,'fold':fold,**metrics(Y[test,j],pred[:,j])})
                    pred_frames.append(frame)
                    print('Baseline',validation,fold,bname,'components',bestn,flush=True)
            for variant in args.variants:
                label=f'{validation} fold={fold}/{args.folds} variant={variant}'
                print('Selecting epochs:',label,flush=True)
                _,_,_,hist,epoch=train(X,Y,W,ids,inner_train,variant,args,device,args.max_epochs,
                                       watch=inner_val,seed=args.seed+fold)
                for rec in hist: history_rows.append({'validation':validation,'fold':fold,'variant':variant,**rec})
                epoch=max(5,epoch)
                # Replay exactly the LR sequence up to the selected inner-validation epoch.
                # No outer-test samples influence model selection or the LR schedule.
                learning_rates=[rec['learning_rate'] for rec in hist[:epoch]]
                print(f'Refit all outer training soils for {epoch} epochs with replayed LR schedule:',label,flush=True)
                net,transform,yscale,_,_=train(X,Y,W,ids,outer_train,variant,args,device,epoch,
                                                watch=None,seed=args.seed+fold,lr_replay=learning_rates)
                pred,_=predict(net,transform,yscale,test,device)
                frame=pd.DataFrame({'row_index':test,'sample_id':ids[test], 'site_id':sites[test],
                                    'water_content':W[test],'time':times[test],'validation':validation,
                                    'fold':fold,'variant':variant,'selected_epochs':epoch})
                for j,t in enumerate(TARGETS):
                    frame[f'true_{t}']=Y[test,j]
                    frame[f'pred_{t}']=pred[:,j]
                pred_frames.append(frame)
                for j,t in enumerate(TARGETS):
                    r=metrics(Y[test,j],pred[:,j])
                    result_rows.append({'target':t,'validation':validation,'variant':variant,'fold':fold,**r})
                # Save partial outputs after each fit, to make failures diagnosable.
                pd.DataFrame(result_rows).to_csv(out/'fold_metrics.csv',index=False)
                pd.concat(pred_frames,ignore_index=True).to_csv(out/'predictions.csv',index=False)
                pd.DataFrame(history_rows).to_csv(out/'inner_learning_curves.csv',index=False)
                del net
                if device.type=='cuda': torch.cuda.empty_cache()
    pred_df=pd.concat(pred_frames,ignore_index=True)
    grouped=[]; stability=[]; bins=[]
    for (validation,variant),df in pred_df.groupby(['validation','variant'],sort=False):
        assert df['row_index'].nunique()==len(X),f'Incomplete test coverage for {validation}/{variant}'
        wc_edges=np.unique(np.quantile(df.water_content,[0,.25,.5,.75,1]))
        wc_bins=pd.cut(df.water_content,wc_edges,include_lowest=True,duplicates='drop')
        for t in TARGETS:
            y=df[f'true_{t}'].to_numpy(float);p=df[f'pred_{t}'].to_numpy(float)
            grouped.append({'target':t,'validation':validation,'variant':variant,**metrics(y,p)})
            abs_err=np.abs(y-p)
            for moisture_bin in wc_bins.cat.categories:
                take=(wc_bins==moisture_bin).to_numpy()
                bins.append({'target':t,'validation':validation,'variant':variant,'moisture_bin':str(moisture_bin),
                             'n':int(np.isfinite(abs_err[take]).sum()),'mae':float(np.nanmean(abs_err[take]))})
            for sample_id,sd in df.assign(_pred=p,_true=y).groupby('sample_id'):
                pr=sd._pred.to_numpy(float); gold=sd._true.to_numpy(float)
                if len(pr)>1 and np.isfinite(pr).sum()>1:
                    stability.append({'target':t,'validation':validation,'variant':variant,'sample_id':sample_id,
                        'n':len(pr),'within_sample_prediction_sd':float(np.nanstd(pr,ddof=1)),
                        'sample_bias':float(np.nanmean(pr-gold)), 'sample_mae':float(np.nanmean(abs(pr-gold)))})
    pd.DataFrame(grouped).to_csv(out/'overall_metrics.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'within_sample_stability.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'moisture_stratified_metrics.csv',index=False)
    # Moisture-response per soil: slope of prediction vs measured water content.
    slopes=[]
    for (validation,variant,sample),sd in pred_df.groupby(['validation','variant','sample_id']):
        if len(sd)<3 or np.ptp(sd.water_content.to_numpy(float))<1e-8:continue
        wc=sd.water_content.to_numpy(float)
        for t in TARGETS:
            pp=sd[f'pred_{t}'].to_numpy(float);yy=sd[f'true_{t}'].to_numpy(float)
            valid=np.isfinite(pp)&np.isfinite(yy)
            if valid.sum()<3 or np.ptp(wc[valid])<1e-8:continue
            slopes.append(dict(validation=validation,variant=variant,sample_id=sample,target=t,n=int(valid.sum()),
                moisture_span=float(np.ptp(wc[valid])),prediction_slope_per_wc=float(np.polyfit(wc[valid],pp[valid],1)[0]),
                mae=float(np.mean(np.abs(pp[valid]-yy[valid])))))
    pd.DataFrame(slopes).to_csv(out/'within_soil_moisture_slopes.csv',index=False)
    # Save split audit; random-split observation-level tests are explicitly diagnostic only.
    audit=[]
    for (validation,fold),sub in pred_df.groupby(['validation','fold']):
        audit.append(dict(validation=validation,fold=int(fold),n_test_observations=int(sub.row_index.nunique()),
                          n_test_soils=int(sub.sample_id.nunique())))
    pd.DataFrame(audit).to_csv(out/'split_summary.csv',index=False)
    pd.DataFrame(split_audit).to_csv(out/'stratification_balance.csv',index=False)
    for v in args.validations:
        subset=pd.DataFrame(grouped)
        sub=subset[subset.validation==v]
        if sub.empty:continue
        plt.figure(figsize=(12,5))
        for variant in (args.variants+(BASELINES if args.baselines else [])):
            part=sub.set_index(['variant','target']).loc[variant].reindex(TARGETS)
            plt.plot(TARGETS,part.r2.values,marker='o',label=variant)
        plt.axhline(0,ls='--',color='gray',lw=.7)
        plt.ylabel('Out-of-fold R²');plt.xticks(rotation=45,ha='right');plt.legend();plt.tight_layout()
        plt.savefig(out/f'r2_{v}.png',dpi=160);plt.close()
    print('\nOverall R²:')
    print(pd.DataFrame(grouped).pivot_table(index=['validation','target'],columns='variant',values='r2').round(3).to_string())
    print('\nSaved outputs:',out,flush=True)

if __name__=='__main__':main()