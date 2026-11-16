"""Mechanism-based proxies; thresholds/windows fixed before unrestricted fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/'output'
SOURCE=Path('D:/OpenAlex/OpenAlex_Analysis/AI subfield capability/data/field_monthly_scores.csv')
SPECS={
 'gain3_l0':('gain',3,0), 'gain6_l0':('gain',6,0), 'gain12_l0':('gain',12,0),
 'gain3_l6':('gain',3,6), 'gain6_l6':('gain',6,6), 'gain12_l6':('gain',12,6),
 'gain6_l12':('gain',6,12),
 'jump5_6_l0':('jump',.05,0), 'jump10_6_l0':('jump',.10,0),
 'jump5_6_l6':('jump',.05,6), 'jump10_6_l6':('jump',.10,6),
 'frontier_l0':('level',0,0), 'frontier_l6':('level',0,6),
 'high70_l0':('high',.70,0), 'high80_l0':('high',.80,0), 'high90_l0':('high',.90,0),
 'high80_l6':('high',.80,6), 'cross80_6_l0':('cross',.80,0),
 'headroom_gain6_l0':('headroom',6,0)
}
ALIASES={'operational_mixed_protocols':'mix','protocol_d0a314e9a4fc':'pd0a','protocol_cca0675e5e29':'pcca'}

def month(s):return pd.Timestamp(s).year*12+pd.Timestamp(s).month-1-1960*12

def build():
    s=pd.read_csv(SOURCE);s['mo']=s.month.map(month)
    assert not s.duplicated(['series','field','mo']).any()
    s.loc[s.status.ne('observed_model_window'),'frontier']=np.nan
    rows=[]
    for (series,field),g in s.groupby(['series','field']):
        lookup=dict(zip(g.mo,g.frontier))
        for mo in range(month('2023-01-01'),month('2026-09-01')+1):
            r=dict(protocol=ALIASES[series],field=field,mo=mo)
            for name,(kind,k,L) in SPECS.items():
                end=mo-L;v=lookup.get(end,np.nan);value=np.nan
                if kind=='level':value=v
                elif kind=='high':value=float(v>=k-1e-12) if np.isfinite(v) else np.nan
                else:
                    length=6 if kind in ('jump','cross') else int(k)
                    path=np.array([lookup.get(x,np.nan) for x in range(end-length,end+1)])
                    if np.isfinite(path).all():
                        if kind=='gain':value=path[-1]-path[0]
                        elif kind=='jump':
                            dx=np.diff(path);value=dx[dx>=k-1e-12].sum()
                        elif kind=='cross':value=float(path[0]<k-1e-12 and path[-1]>=k-1e-12)
                        elif kind=='headroom' and path[0]<1-1e-12:value=(path[-1]-path[0])/(1-path[0])
                r[name]=value
            rows.append(r)
    out=pd.DataFrame(rows);out.to_csv(OUT/'field_month_proxies.csv',index=False)
    (OUT/'proxy_specifications.json').write_text(json.dumps(dict(specs=SPECS,
      primary=['gain6_l0','gain6_l6'],core_protocol='mix',core_mapping='original',
      units='All levels/gains are fractions 0-1; thresholds are indicators. Six-month gain is F(t-L)-F(t-L-6).',
      jump_definition='Sum monthly positive increments >=5 or >=10 percentage points in a six-month window, with full history.',
      cross_definition='Known crossing of 80% inside an observed six-month window; not an invented historical first crossing.',
      multiplicity='Holm for two primary tests; BH across 19 original/mix own-sample proxies; alternate mappings are sensitivity, not independent replications.',
      missing='No zero filling or extrapolation; require complete history for every gain window.'),indent=2),encoding='utf-8')
    return out

if __name__=='__main__':build()
