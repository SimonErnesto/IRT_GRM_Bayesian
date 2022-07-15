# -*- coding: utf-8 -*-
import os
import pandas as pd
import pymc3 as pm
import numpy as np
from scipy.special import expit
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

############# CHS BSEM model analysis ###############


os.chdir("")

data = pd.read_csv("/data/chs_data_clean.csv")

#data = data.sort_values('item')

scores = []
for s in range(len(data)):
    if data.qdirection[s] == 'reversed':
        scores.append(4 - data.score[s])
    if data.qdirection[s] == 'forward':
        scores.append(data.score[s])

data['score'] = scores
    

subjects = data.subject.unique() #subjects id
items = data.item.unique() #items names

sub_lookup = dict(zip(data.subject.unique(), range(len(data.subject.unique()))))
sub_s = data.subject.replace(sub_lookup).values #subject index

item_lookup = dict(zip(data.item.unique(), range(len(data.item.unique()))))
item_i = data.item.replace(item_lookup).values #item index


responses = data.score.values
C = len(data.score.unique()) #Number of categories (unique scores)
I = len(data.item.unique()) #Number of items
D = len(data.dimension.unique()) #Number of participants/subjects
S = len(data.subject.unique()) #Number of participants/subjects

mat_dot = pm.math.matrix_dot


with pm.Model() as mod:
    
    #delta = pm.HalfNormal('delta', sigma=0.5, shape=(I), testval=np.ones(I)) #discrimination parameter
    delta = pm.Lognormal('delta', mu=0, sigma=0.5, shape=(I), testval=np.ones(I)) #discrimination parameter

    psi = pm.Normal('psi', mu=0, sigma=0.5, shape=S, testval=np.ones(S)) #participant/subject parameter
    
    ksigma = pm.HalfNormal('ksigma', 0.5)
    kmu = pm.Normal('kmu', [-4,-2,2,4], 0.5, shape=C-1)
    kappa = pm.Normal('kappa', mu=kmu, sigma=ksigma, transform=pm.distributions.transforms.ordered,
                      shape=(I,C-1), testval=np.arange(C-1)) #cutpoints/difficulty parameter 
    
    eta = delta[item_i]*psi[sub_s] #estimate
    
    y = pm.OrderedLogistic('y', cutpoints=kappa[item_i], eta=eta, observed=responses)


######### Prio Predictive Checks ################    
os.chdir("/prior_preds/")

with mod:
    preds = pm.sample_prior_predictive(samples=1000,
        var_names=["psi", "kappa", "delta"])

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

for i in tqdm(range(I)):
    num = items[i].split('_')[1]
    num = num.replace('q','')
    if 'q'+str(i) == 'q'+str(num):
        sco = data[data.item==items[i]]['score'].values
    if 'blocking' in items[i]:
        name = 'Item '+str(num)+' (Blocking): '
        color1 = 'firebrick'
        color2 = 'crimson'
        d = 0
    if 'hiding' in items[i]:
        name = 'Item '+str(num)+' (Hiding): '
        color1 = 'navy'
        color2 = 'mediumblue'
        d = 1
    if 'inspecting' in items[i]:
        name = 'Item '+str(num)+' (Inspecting): '
        color1 = 'goldenrod'
        color2 = 'gold'
        d = 2
    cuts = preds['kappa'][:,i].T
    prob = cuts - preds['psi'].mean(axis=1)*preds['delta'][:,i]
    posts = np.array([pordlog(prob.T[s]) for s in range(len(prob.T))]).T
    prop = data[data.item==items[i]]
    prop = [len(prop[prop.score==s])/len(prop) for s in [0,1,2,3,4]]
    question = data[data.item==items[i]].question.values[0].replace('#', '')
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    plt.plot(pmeans, color=color1, linewidth=2, label='Predictions Mean')
    plt.fill_between([0,1,2,3,4],h5s,h95s, color=color2, alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.suptitle(name+'Prior Probability')
    plt.title(question.replace('.',''), size=11)
    plt.grid(alpha=0.1)
    plt.legend(prop={'size': 10})
    plt.xticks(range(0,5))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(items[i]+'_prob.png', dpi=300)
    plt.close()
    
    
########### Sample Model ################
os.chdir("")

tracedir = "/trace/"
with mod:
    trace = pm.sample(2000, tune=2000, chains=4, cores=16, target_accept=0.95)
    #trace = pm.load_trace(tracedir)
   
pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)
 

pm.model_to_graphviz(mod).render('mod_graph')

az.plot_energy(trace)
plt.savefig('energy.png',dpi=100)
plt.close()

summ = az.summary(trace, hdi_prob=0.9, round_to=4)
summ.to_csv('summary.csv')


with mod:
    ppc = pm.sample_posterior_predictive(trace)
fig, ax = plt.subplots()
samps = np.random.randint(0,1000,100)
preds = np.array(ppc['y'])
for s in samps[:-1]:
    ax = sns.kdeplot(preds[s], gridsize=1000, color=(0.2, 0.6, 0.5, 0.2))
sns.kdeplot(preds[samps[-1]], gridsize=1000, color=(0.2, 0.6, 0.5, 0.3), label='Predicted Responses')
sns.kdeplot(responses, gridsize=1000, color='darkviolet', label='Observed Responses')
plt.xlabel('Ratings (scores)')
plt.title('Posterior Predictive Checks')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('ppc.png',dpi=300)
plt.close()
plt.close()


######### Posterior probabilities ################    
os.chdir("/response_prob/")

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

for i in tqdm(range(I)):
    num = items[i].split('_')[1]
    num = num.replace('q','')
    if 'q'+str(i) == 'q'+str(num):
        sco = data[data.item==items[i]]['score'].values
    if 'blocking' in items[i]:
        name = 'Item '+str(num)+' (Blocking): '
        color1 = 'firebrick'
        color2 = 'crimson'
        d = 0
    if 'hiding' in items[i]:
        name = 'Item '+str(num)+' (Hiding): '
        color1 = 'navy'
        color2 = 'mediumblue'
        d = 1
    if 'inspecting' in items[i]:
        name = 'Item '+str(num)+' (Inspecting): '
        color1 = 'goldenrod'
        color2 = 'gold'
        d = 2
    cuts = trace['kappa'][:,i].T
    prob = cuts - trace['psi'].mean(axis=1)*trace['delta'][:,i]
    posts = np.array([pordlog(prob.T[s]) for s in range(len(prob.T))]).T
    prop = data[data.item==items[i]]
    prop = [len(prop[prop.score==s])/len(prop) for s in [0,1,2,3,4]]
    question = data[data.item==items[i]].question.values[0].replace('#', '')
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    plt.plot(pmeans, color=color1, linewidth=2, label='Predictions Mean')
    plt.fill_between([0,1,2,3,4],h5s,h95s, color=color2, alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.suptitle(name+'Prior Probability')
    plt.title(question.replace('.',''), size=11)
    plt.grid(alpha=0.1)
    plt.legend(prop={'size': 10})
    plt.xticks(range(0,5))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(items[i]+'_prob.png', dpi=300)
    plt.close()


######### Item responses ################    
os.chdir("/item_characteristics/")

test_char_mean = []
test_char_h5 = []
test_char_h95 = []
colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
lines = ['dotted', 'dashed', 'solid', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]
trait = [trace['psi'][:,s].mean() for s in range(S)]
for i in tqdm(range(I)):
    if 'blocking' in items[i]:
        d = 0
    if 'hiding' in items[i]:
        d = 1
    if 'inspecting' in items[i]:
        d = 2
    question = data[data.item==items[i]].question.values[0].replace('#', '')        
    name1 = 'Item ' + items[i].split('_')[1].replace('q', '')
    name2 = ' ('+items[i].split('_')[0].capitalize()+'):'
    name = name1+name2
    num = items[i].split('_')[1]
    num = num.replace('q','')
    direction = data[data.item==items[i]].qdirection.values[0]
    cuts = trace['kappa'][:,i].T
    if direction == 'forward':
        prob = np.array([cuts - trace['psi'][:,s]*trace['delta'][:,i] for s in range(S)])
    if direction == 'reversed':
        prob = np.array([cuts + trace['psi'][:,s]*trace['delta'][:,i] for s in range(S)])
    prob = np.array([[pordlog(prob[s].T[p]) for p in range(len(prob.T))] for s in range(S)])
    chars = []
    h5s = []
    h95s = []
    for c in range(C):
        char = np.array([prob[s].T[c].mean() for s in range(S)])
        h5 = np.array([az.hdi(prob[s].T[c], hdi_prob=0.9)[0] for s in range(S)])
        h95 = np.array([az.hdi(prob[s].T[c], hdi_prob=0.9)[1] for s in range(S)])
        x, y, y1, y2 = zip(*sorted(zip(trait, char, h5, h95))) 
        plt.plot(x, y, color=colors[c], label='Score: '+str(c), linestyle=lines[c])
        plt.fill_between(x, y1, y2, alpha=0.3, color=colors[c])
        plt.grid(alpha=0.2) 
        plt.suptitle(name+'Item Characteristic Curve')
        plt.title(question.replace('.',''), size=11)
        plt.legend(prop={'size': 10})
        plt.xlabel('Cybersecurity Habits (\u03C8)')
        plt.ylabel('Probability')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        chars.append(char)
        h5s.append(h5)
        h95s.append(h95)
    plt.savefig(items[i]+"_char.png", dpi=300)
    plt.close()
    test_char_mean.append(np.array(chars))
    test_char_h5.append(np.array(h5s))
    test_char_h95.append(np.array(h95s))

###Plot test chars
tm = np.array(test_char_mean).sum(axis=0)
th5 = np.array(test_char_h5).sum(axis=0)
th95 = np.array(test_char_h95).sum(axis=0)
trait = [trace['psi'][:,s].mean() for s in range(S)]
for c in range(C):
    colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
    char = np.array([tm[c][s] for s in range(S)])
    h5 = np.array([th5[c][s] for s in range(S)])
    h95 = np.array([th95[c][s] for s in range(S)])
    x, y, y1, y2 = zip(*sorted(zip(trait, char, h5, h95))) 
    plt.plot(x, y, color=colors[c], label='Score: '+str(c), linestyle=lines[c])
    plt.fill_between(x, y1, y2, alpha=0.3, color=colors[c])
    plt.grid(alpha=0.2) 
    plt.title('Test Characteristic Curve')
    plt.legend(prop={'size': 10})
    plt.xlabel('Cybersecurity Habits (\u03C8)')
    plt.ylabel('Answers to Items')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.savefig("test_char.png", dpi=300)
plt.close()


######### Item Information ################    
os.chdir("/item_information/")

test_info_mean = []
test_info_h5 = []
test_info_h95 = []
colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
lines = ['dotted', 'dashed', 'solid', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]
trait = [trace['psi'][:,s].mean() for s in range(S)]
for i in tqdm(range(I)):
    if 'blocking' in items[i]:
        d = 0
    if 'hiding' in items[i]:
        d = 1
    if 'inspecting' in items[i]:
        d = 2
    direction = data[data.item==items[i]].qdirection.values[0]
    cuts = trace['kappa'][:,i].T
    if direction == 'forward':
        prob = np.array([cuts - trace['psi'][:,s]*trace['delta'][:,i] for s in range(S)])
    if direction == 'reversed':
        prob = np.array([cuts + trace['psi'][:,s]*trace['delta'][:,i] for s in range(S)])
    prob = np.array([[pordlog(prob[s].T[p]) for p in range(len(prob.T))] for s in range(S)])
    question = data[data.item==items[i]].question.values[0].replace('#', '')        
    name1 = 'Item ' + items[i].split('_')[1].replace('q', '')
    name2 = ' ('+items[i].split('_')[0].capitalize()+'):'
    name = name1+name2
    num = items[i].split('_')[1]
    num = num.replace('q','')
    infos = []
    h5s = []
    h95s = []
    for c in range(C):
        colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
        probs = np.array([(prob.T[c][:,s]*(1-prob.T[c][:,s]))*(trace['delta'][:,i]**2) for s in range(S)])
        info = np.array([probs[s].mean() for s in range(S)])
        h5 = np.array([az.hdi(probs[s], hdi_prob=0.9)[0] for s in range(S)])
        h95 = np.array([az.hdi(probs[s], hdi_prob=0.9)[1] for s in range(S)])
        x, y, y1, y2 = zip(*sorted(zip(trait, info, h5, h95))) 
        plt.plot(x, y, color=colors[c], label='Score: '+str(c), linestyle=lines[c])
        plt.fill_between(x, y1, y2, alpha=0.3, color=colors[c])
        plt.grid(alpha=0.2) 
        plt.suptitle(name+'Item Information Curve')
        plt.title(question.replace('.',''), size=11)
        plt.legend(prop={'size': 10})
        plt.xlabel('Cybersecurity Habits (\u03C8)')
        plt.ylabel('Information')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        infos.append(info)
        h5s.append(h5)
        h95s.append(h95)
    plt.savefig(items[i]+"_info.png", dpi=300)
    plt.close()
    test_info_mean.append(np.array(infos))
    test_info_h5.append(np.array(h5s))
    test_info_h95.append(np.array(h95s))

###Plot test infos
tm = np.array(test_info_mean).sum(axis=0)
th5 = np.array(test_info_h5).sum(axis=0)
th95 = np.array(test_info_h95).sum(axis=0)
trait = [trace['psi'][:,s].mean() for s in range(S)]
for c in range(C):
    colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
    info = np.array([tm[c][s] for s in range(S)])
    h5 = np.array([th5[c][s] for s in range(S)])
    h95 = np.array([th95[c][s] for s in range(S)])
    x, y, y1, y2 = zip(*sorted(zip(trait, info, h5, h95))) 
    print(max(y))
    ma = np.argmax(y)
    print(x[ma])
    plt.plot(x, y, color=colors[c], label='Score: '+str(c), linestyle=lines[c])
    plt.fill_between(x, y1, y2, alpha=0.3, color=colors[c])
    plt.grid(alpha=0.2) 
    plt.title('Test Information Curve')
    plt.legend(prop={'size': 10})
    plt.xlabel('Cybersecurity Habits (\u03C8)')
    plt.ylabel('Information')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.savefig("test_info.png", dpi=300)
plt.close()



os.chdir("")
from matplotlib.ticker import FormatStrFormatter

# Reliability π=1/(1+1/I).
delta = trace['delta']
theta = np.array([trace['kappa'][:,i].T - trace['psi'].mean(axis=1)*trace['delta'][:,i] for i in range(I)])
p = np.array([[pordlog(theta[i,:,x]) for x in range(len(theta.T))] for i in range(I)])
Info = np.array([[(delta[:,i]**2)*p[i,:,c]*(1-p[i,:,c]) for i in range(I)] for c in range(C)])
Info = Info.sum(axis=1)
pi = np.array([1/(1+(1/Info[c])) for c in range(C)])
colors = ['forestgreen', 'firebrick', 'steelblue', 'goldenrod', 'darkviolet']
n=0
for c in range(C):
    n += 2
    color=colors[c]
    m = np.median(pi[c]).round(3)
    h5,h95 = az.hdi(pi[c], hdi_prob=0.9)
    h5 = round(h5, 3)
    h95 = round(h95, 3)
    x = [h5,m,h95]
    y = [n,n,n]
    plt.plot(x,y, color=color)
    plt.plot(m,n,'o',color='white', markeredgecolor=color)
    plt.plot(h5,n,'|',color=color, markeredgewidth=2)
    plt.plot(h95,n,'|',color=color, markeredgewidth=2)
    plt.xlabel('Reliability ($π_{c}$)')
    plt.ylabel('Score')
    plt.grid(alpha=0.2) 
    plt.yticks(ticks=[2,4,6,8,10], labels=['0','1','2','3','4'])
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.title('Test Reliability per Score')
plt.savefig('reliability_score_posteriors.png', dpi=300)




rho = 1/(1 + ((trace['delta']**2).sum(axis=1))**-1)
az.plot_posterior(rho, hdi_prob=0.9, kind='hist', color='m', round_to=3)
plt.title('Reliability ($ρ$)')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig('reliability_posterior.png', dpi=300)