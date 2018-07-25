
# coding: utf-8

# In[19]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import markov_clustering as mc
import networkx as nx
import random
import numpy as np
import os


# In[11]:


def make_graph(sim, labels=None):
    G = nx.Graph()
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if i != j and sim[i,j] != 0:
                if labels == None:
                    G.add_edge(i, j, weight=sim[i,j])
                else:
                    G.add_edge(labels[i], labels[j], weight=sim[i,j])
    return G


# In[12]:


class Config():
    colors = ['aquamarine', 'bisque', 'blanchedalmond', 'blueviolet', 'brown',
              'burlywood', 'cadetblue', 'chartreuse','chocolate', 'coral',
              'cornflowerblue', 'cornsilk', 'crimson', 'darkblue', 'darkcyan',
              'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
              'darkmagenta', 'darkolivegreen', 'darkorange', 'darkslateblue',
              'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
              'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
              'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue',
              'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
              'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
              'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory']
    labels = None


# In[13]:


simfilename = "E:\sws\sim.tsv"
labelfilename = "E:\sws\labels.csv"

X = np.genfromtxt(simfilename, delimiter=' ', dtype=None)
label2idx = dict()
sim = np.ones((len(X), len(X)))
if os.path.exists(labelfilename):
    os.remove(labelfilename)
f = open(labelfilename, 'w')
for t in X:
    t0, t1, t2 = t
    if not t0 in label2idx:
        idx = len(label2idx)
        label2idx[t0] = idx
        f.write(str(idx) + "," + str(t0) + "\n")
    if not t1 in label2idx:
        idx = len(label2idx)
        label2idx[t1] = len(label2idx)
        f.write(str(idx) + "," + str(t1) + "\n")
    idx0 = label2idx[t0]
    idx1 = label2idx[t1]
    sim[idx0][idx1] = t2
    sim[idx1][idx0] = t2
f.close()

print("Restored {}x{} matrix".format(sim.shape[0], sim.shape[1]))


# In[20]:


Config.labels = []
with open('E:\sws\labels.csv') as f:
    for line in f:
        _, label = line.rstrip().split(",")
        Config.labels.append(label)

print("Loaded labels (" + str(len(Config.labels)) + " classes): ", end='')
print(Config.labels)


# In[93]:


threshold = 0.75
adjmat = sim.reshape((-1,)).copy()
adjmat[adjmat > threshold] = 0
#adjmat[adjmat > 0] = 1
print("{} out of {} values set to zero".format(len(adjmat[adjmat == 0]), len(adjmat)))
adjmat = adjmat.reshape(sim.shape)


# In[94]:


G = make_graph(adjmat, labels=Config.labels)
nx.draw_spring(G, with_labels=True)


# In[95]:


matrix = nx.to_scipy_sparse_matrix(G)
result = mc.run_mcl(matrix, inflation=2)           # run MCL with default parameters
clusters = mc.get_clusters(result)    # get clusters
print("There are {} clusters.".format(len(clusters)))
mc.draw_graph(matrix, clusters, with_labels=True, edge_color="silver")  


# In[77]:


ref = np.genfromtxt(labelfilename, delimiter=',', dtype=None)
print(ref[19])

