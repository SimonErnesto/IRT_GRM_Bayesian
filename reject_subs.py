# -*- coding: utf-8 -*-
import os
import pandas as pd

data = pd.read_csv("/data/chs_data.csv")

os.chdir("/data/")

data = data[data['Reaction Time'] > 0]

data = data.dropna(axis=0, subset=['Response'])

data = data[data['display'] == 'chs']

subs = data['Participant Private ID'].unique()

check = data[data['dimension'] == 'checkup']


rejected_subs = []
for s in range(len(pd.unique(subs))):
    reject = []
    sub = check[check['Participant Private ID'] == subs[s]]
    for q in sub.qnum.unique():
        if sub[sub.qnum==q]['qdirection'].values[0] == 'reversed':
           if sub[sub.qnum==q]['Response'].values[0] > 1:
               reject.append('reject')
        if sub[sub.qnum==q]['qdirection'].values[0] == 'forward':       
            if sub[sub.qnum==q]['Response'].values[0] < 4:
                reject.append('reject')
    print(reject)
    if len(reject) > 2:
        rejected_subs.append(subs[s])

for r in rejected_subs:
    re = check[check['Participant Private ID']==r] 
    re.to_csv('reject'+str(r)+'.csv')