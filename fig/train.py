import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from sklearn import manifold
from collections import OrderedDict
import os
import pandas as pd

import torch

perplexity = 90
topk = 5
vol = 'small'
MODE = 'train'

num_classes=5
color = ['red', 'green', 'blue', 'orange', 'brown']
per_class = None
per_ood = None
RANGE_OOD = None
RANGE_class = None

if MODE == 'all':
    RANGE_OOD = 1200
    RANGE_class = 150
    per_ood = 300
    per_class = 150
elif MODE == 'train':
    RANGE_OOD = 100
    RANGE_class = 100
    per_ood = 100
    per_class = 100
elif MODE == 'valid':
    RANGE_OOD = 100
    RANGE_class = 20
    per_ood = 100
    per_class = 20
elif MODE == 'test':
    RANGE_OOD = 1000
    RANGE_class = 30
    per_ood = 300
    per_class = 30

# Train: 100*150 + 100ood = 15100
# Valid: 100ood  + 20*150 = 3100
# Test:  1000ood + 30*150 = 5500
# Total: 1200ood + 150*150 = 23700

filepath = '../supervised-contrastive-learning-OOD/pts/'
X_train_naive = (torch.load(filepath+ 'train_naive.pt', map_location=torch.device('cpu'))).detach().numpy()
X_valid_naive = (torch.load(filepath+ 'valid_naive.pt', map_location=torch.device('cpu'))).detach().numpy()
X_test_naive = (torch.load(filepath+ 'test_naive.pt', map_location=torch.device('cpu'))).detach().numpy()
X_train_cl = (torch.load(filepath+ 'train_cl.pt', map_location=torch.device('cpu'))).detach().numpy()
X_valid_cl = (torch.load(filepath+ 'valid_cl.pt', map_location=torch.device('cpu'))).detach().numpy()
X_test_cl = (torch.load(filepath+ 'test_cl.pt', map_location=torch.device('cpu'))).detach().numpy()


filepath = '../knn/pts/'
X_train_knn = (torch.load(filepath+ 'knn_train' + str(topk) + '.pt', map_location=torch.device('cpu'))).detach().numpy()
X_valid_knn = (torch.load(filepath+ 'knn_val' + str(topk) + '.pt', map_location=torch.device('cpu'))).detach().numpy()
X_test_knn = (torch.load(filepath+'knn_test' + str(topk) + '.pt', map_location=torch.device('cpu'))).detach().numpy()


file_name = '../knn/CLINC/train.csv'
df = pd.read_csv(file_name, sep='\t')
labels = np.array(df['label'])
y = labels[np.arange(0, len(labels), len(labels)//150)]


Xs = OrderedDict()
Xs['naive'] = OrderedDict()
Xs['cl'] = OrderedDict()
Xs['knn'] = OrderedDict()
OOD_naive = None
OOD_cl = None
OOD_knn = None
if MODE == 'all':
    OOD_naive = (np.concatenate([X_train_naive[15000:], X_valid_naive[:100], X_test_naive[:1000]]))
    OOD_cl = (np.concatenate([X_train_cl[15000:], X_valid_cl[:100], X_test_cl[:1000]]))
    OOD_knn = (np.concatenate([X_train_knn[15000:], X_valid_knn[:100], X_test_knn[:1000]]))
elif MODE == 'train':
    OOD_naive = (np.concatenate([X_train_naive[15000:]]))
    OOD_cl = (np.concatenate([X_train_cl[15000:]]))
    OOD_knn = (np.concatenate([X_train_knn[15000:]]))
elif MODE == 'valid':
    OOD_naive = (np.concatenate([X_valid_naive[:100]]))
    OOD_cl = (np.concatenate([X_valid_cl[:100]]))
    OOD_knn = (np.concatenate([X_valid_knn[:100]]))
elif MODE == 'test':
    OOD_naive = (np.concatenate([X_test_naive[:1000]]))
    OOD_cl = (np.concatenate([X_test_cl[:1000]]))
    OOD_knn = (np.concatenate([X_test_knn][:1000]))


class_indices = np.arange(150)
#np.random.shuffle(class_indices)
fig_labels = ['Original Bert', 'Supervised Contrastive Learning', 'Ours(Topk='+str(topk)+')']
epochs = 150 // num_classes
assert(epochs * num_classes == 150)
for epoch in range(epochs):
    Xs['naive']['class'] = []
    Xs['cl']['class'] = []
    Xs['knn']['class'] = []
    chosen_ood = np.random.choice(np.arange(RANGE_OOD), per_ood, replace=False)
    Xs['naive']['OOD'] = OOD_naive[chosen_ood]
    Xs['cl']['OOD'] = OOD_cl[chosen_ood]
    Xs['knn']['OOD'] = OOD_knn[chosen_ood]
    for i in range(num_classes):
        id = class_indices[epoch*num_classes + i]
        if MODE == 'all':
            embed1 = np.concatenate([X_train_naive[id*100:(id+1)*100], X_valid_naive[100+20*id:100+20*(id+1)], X_test_naive[1000+30*id:1000+30*(id+1)]])
            embed2 = np.concatenate([X_train_cl[id*100:(id+1)*100], X_valid_cl[100+20*id:100+20*(id+1)], X_test_cl[1000+30*id:1000+30*(id+1)]])
            embed3 = np.concatenate([X_train_knn[id*100:(id+1)*100], X_valid_knn[100+20*id:100+20*(id+1)], X_test_knn[1000+30*id:1000+30*(id+1)]])
        elif MODE == 'train':
            embed1 = X_train_naive[id*100:(id+1)*100]
            embed2 = X_train_cl[id*100:(id+1)*100]
            embed3 = X_train_knn[id*100:(id+1)*100]
        elif MODE == 'valid':
            embed1 = X_valid_naive[100+20*id:100+20*(id+1)]
            embed2 = X_valid_cl[100+20*id:100+20*(id+1)]
            embed3 = X_valid_knn[100+20*id:100+20*(id+1)]
        elif MODE == 'test':
            embed1 = X_test_naive[1000+30*id:1000+30*(id+1)]
            embed2 = X_test_cl[1000+30*id:1000+30*(id+1)]
            embed3 = X_test_knn[1000+30*id:1000+30*(id+1)]
        chosen_samp = np.random.choice(np.arange(RANGE_class), per_class, replace=False)
        Xs['naive']['class'].append(embed1[chosen_samp])
        Xs['cl']['class'].append(embed2[chosen_samp])
        Xs['knn']['class'].append(embed3[chosen_samp])

    (fig, subplots) = plt.subplots(1, 3, figsize=(20, 8))
    for i, (method, X)in enumerate(Xs.items()):

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
        X = np.concatenate([X['OOD'], np.concatenate(X['class'])])
        Y = tsne.fit_transform(X)

        ax = subplots[i]

        for j in range(num_classes):
            C = Y[per_ood+j*per_class:per_ood+(j+1)*per_class]
            ax.scatter(C[:, 0], C[:, 1], s=30, c=color[j], alpha=0.5, marker='o', label=y[class_indices[num_classes*epoch+j]])

        OOD = Y[:per_ood]
        ax.scatter(OOD[:, 0], OOD[:, 1], s=30, c='black', alpha=0.5, marker='o', label='ood')
        ax.legend()
        ax.set_title("%s" % fig_labels[i])
        ax.set_facecolor('whitesmoke')
        ax.grid(linewidth=0.5, color='black', alpha=0.1)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
    plt.subplots_adjust(wspace=0.05, hspace=0)

    dir1 = MODE + '_' + str(topk) + '_' + str(per_ood) + '_' + str(perplexity)
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    start = num_classes*epoch
    end = num_classes*(epoch+1)-1
    filename = dir1 + '/' + str(start) + '-' + str(end) + '.jpeg'
    plt.savefig(filename)
    plt.close()
    print('%s is saved' % filename)