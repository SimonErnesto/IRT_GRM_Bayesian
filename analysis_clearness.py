# -*- coding: utf-8 -*-
import os
import pandas as pd
import pymc3 as pm
import numpy as np
from scipy.special import expit
import arviz as az
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

data = pd.read_csv("data/chs_clearness_clean.csv")

score = data.score

subjects = data.subject.unique() #subjects id
items = data.item.unique() #items names

sub_lookup = dict(zip(data.subject.unique(), range(len(data.subject.unique()))))
sub_s = data.subject.replace(sub_lookup).values #subject index

item_lookup = dict(zip(data.item.unique(), range(len(data.item.unique()))))
item_i = data.item.replace(item_lookup).values #item index

I = len(data.item.unique())
S = len(sub_lookup)
C = len(data.score.unique())

with pm.Model() as mod:
    e = pm.Normal('e', 0, 1, shape=I)
    c = pm.Normal('c', mu=0, sigma=1, transform=pm.distributions.transforms.ordered,
                      shape=(C-1), testval=np.arange(C-1))  
    y = pm.OrderedLogistic('y', cutpoints=c, eta=e[item_i], observed=score)
    
with mod:
    trace = pm.sample()

tracedir = "/trace/"
pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)


#### Plot response probability #####

os.chdir("/response_prob/")

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

props = []
for i in tqdm(range(I)):
    cuts = trace['c'].T
    num = items[i].split('_')[1]
    num = num.replace('q','')
    prob = cuts - trace['e'][:,i]
    posts = np.array([pordlog(prob.T[s]) for s in range(len(prob.T))]).T
    prop = data[data.item==items[i]]
    prop = [len(prop[prop.score==s])/len(prop) for s in [0,1,2]]
    props.append(prop)
    question = data[data.item==items[i]].question.values[0].replace('#', '')
    if 'q'+str(i) == 'q'+str(num):
        sco = data[data.item==items[i]]['score'].values
    if 'blocking' in items[i]:
        name = 'Item '+str(num)+' (Blocking): '
        color1 = 'firebrick'
        color2 = 'crimson'
    if 'hiding' in items[i]:
        name = 'Item '+str(num)+' (Hiding): '
        color1 = 'navy'
        color2 = 'mediumblue'
    if 'inspecting' in items[i]:
        name = 'Item '+str(num)+' (Inspecting): '
        color1 = 'goldenrod'
        color2 = 'gold'
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    plt.plot(pmeans, color=color1, linewidth=2, label='Posterior Mean')
    plt.fill_between([0,1,2],h5s,h95s, color=color2, alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.suptitle(name+'Prior Probability')
    plt.title(question.replace('.',''), size=11)
    plt.grid(alpha=0.1)
    plt.xticks(range(0,3))
    plt.legend(prop={'size': 10})
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(items[i]+'_prob.png', dpi=300)
    plt.close()

summ = az.summary(trace, hdi_prob=0.9)
summ.to_csv('summary.csv')


################# Participant Model #####################

with pm.Model() as mod:
    e = pm.Normal('e', 0, 1, shape=S)
    c = pm.Normal('c', mu=0, sigma=1, transform=pm.distributions.transforms.ordered,
                      shape=(C-1), testval=np.arange(C-1))  
    y = pm.OrderedLogistic('y', cutpoints=c, eta=e[sub_s], observed=score)
    
with mod:
    trace = pm.sample()
    #trace = pm.load_trace(tracedir)
    

# pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)


#### Plot response probability #####
os.chdir("/response_prob_sub/")

color = 'forestgreen'

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]
props = []
for s in tqdm(range(S)):
    cuts = trace['c'].T
    name = subjects[s]
    prob = cuts - trace['e'][:,s]
    posts = np.array([pordlog(prob.T[p]) for p in range(len(prob.T))]).T
    prop = data[data.subject==subjects[s]]
    prop = [len(prop[prop.score==p])/len(prop) for p in [0,1,2]]
    props.append(prop)
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    plt.plot(pmeans, color=color, linewidth=2, label='Posterior Mean')
    plt.fill_between([0,1,2],h5s,h95s, color=color, alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(name+'Prior Probability')
    plt.grid(alpha=0.1)
    plt.xticks(range(0,3))
    plt.legend(prop={'size': 10})
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(subjects[s]+'_prob.png', dpi=300)
    plt.close()
    
    

os.chdir("")
summ = az.summary(trace, hdi_prob=0.9)
summ.to_csv('summary_clearness_subject.csv')