# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

data = pd.read_csv("/data/chs_data.csv")

os.chdir("/data/")

data = data[data['Reaction Time'] > 0]

data = data.dropna(axis=0, subset=['Response'])

data = data[data['display'] == 'chs']

data = data[data['dimension'].isin(['blocking', 'hiding', 'inspecting'])]

data.rename(columns = {'Participant Private ID':'subject'}, inplace = True)


data = data[['subject', 'Response', 'display', 'question', 
             'dimension', 'qnum', 'qdirection']]


data.rename(columns={'qnum':'question_number', 'Response':'score'}, inplace = True)

data = data.drop_duplicates(subset=(['subject','question_number']))

data['item'] = data.dimension.str.cat("_"+data.question_number) 

subs = []
n = 0
for s in data.subject.unique():
    d = data[data.subject==s]
    n += 1
    if n < 10:
        name = 'sub00'+str(n)
    if n >= 10 and n < 100:
        name = 'sub0'+str(n)
    if n >= 100:
        name = 'sub'+str(n)
    subname = np.repeat(name, len(d))
    subs.append(subname)
    if len(d) > 44:
        print(name)

data['subject'] = np.concatenate(subs)

data.to_csv('chs_data_clean.csv')

