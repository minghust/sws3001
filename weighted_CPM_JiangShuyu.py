# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:19:38 2018

@author: Administrator
"""

#%%
import networkx as nx
import matplotlib.pyplot as plt
import PIL.ImageOps    
from PIL import Image
import numpy as np
import itertools
import math
import os
from networkx.algorithms.community import k_clique_communities
#%%
def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

# Creates a NetworkX graph object
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
# Save graph for use in Gephi or pals
def export_edge_list(sim, labels=None, filename="edges.csv", delim=",", header=True):
    f = open(filename, 'w')
    if header:
        f.write("Source,Target\n")
    for i in range(sim.shape[0]):
        for j in range(i+1, sim.shape[1]):
            if sim[i,j] != 0:
                if labels == None:
                    f.write(str(i) + delim + str(j) + "\n")
                else:
                    f.write("\"" + labels[i] + "\"" + delim + "\"" + labels[j] + "\"\n")                          
    f.close()
"""##Configuration"""

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


#%%    
# Load a saved copy of the dissimilarity matrix
"""---


## Build graph
Construct an adjacency matrix from the dissimilarity matrix, then use the adjacency matrix to build a networkx graph
"""

# Load a saved copy of the dissimilarity matrix

simfilename = "sim.tsv"
labelfilename = "label.csv"

X = np.genfromtxt(simfilename, delimiter=' ', encoding='utf-8', dtype=None)
#print(X)
label2idx = dict()
sim = np.zeros((250, 250))
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
        f.write(str(idx) + "," +  str(t1) + "\n")
    idx0 = label2idx[t0]
    idx1 = label2idx[t1]
    sim[idx0][idx1] = t2
    sim[idx1][idx0] = t2


f.close()

Config.labels = []
with open(labelfilename) as f:
    for line in f:
        _id, label = line.rstrip().split(",")
        _type = str(int((int(_id)/10)))+'-'+str(int(_id)%10)
        Config.labels.append(_type)

#print("Loaded labels (" + str(len(Config.labels)) + " classes): ", end='')
#print(Config.labels)
#%%
# Analyze distribution of dissimilarity score

simflat = sim.reshape((-1,))
simflat = simflat[simflat != 0] # Too many ones result in a bad histogram so we remove them
# _ = plt.hist(simflat, bins=25)

mmax  = np.max(simflat)
mmin  = np.min(simflat)
mmean = np.mean(simflat)
# print('avg={0:.2f} min={1:.2f} max={2:.2f}'.format(mmean, mmin, mmax))

# Select a suitable threshold and set dissimilarity scores larger than that threshold to zero

threshold = 0.7
adjmat = sim.copy()
np.fill_diagonal(adjmat, np.min(sim)) # Set the diagonal elements to a small value so that they won't be zeroed out
adjmat = adjmat.reshape((-1,))
adjmat[adjmat > threshold] = 0
#print(adjmat)
# print("{} out of {} values set to zero".format(len(adjmat[adjmat == 0]), len(adjmat)))
adjmat = adjmat.reshape(sim.shape)

# Construct a networkx graph from the adjacency matrix
# (Singleton nodes are excluded from the graph)
G = make_graph(adjmat, labels=Config.labels)
# nx.draw(G, with_labels=True)


#%%
from collections import defaultdict

import networkx as nx
class CPM():
    def __init__(self,G,k=4):

        self._G = G

        self._k = k
    def execute(self):

        # find all cliques which size > k

        cliques = list(nx.find_cliques(G))

        vid_cid = defaultdict(lambda:set())

        for i,c in enumerate(cliques):

            if len(c) < self._k:

                continue

            for v in c:

                vid_cid[v].add(i)
        # build clique neighbor

        clique_neighbor = defaultdict(lambda:set())

        remained = set()

        for i,c1 in enumerate(cliques):

            #if i % 100 == 0:

                #print i

            if len(c1) < self._k:

                continue

            remained.add(i)

            s1 = set(c1)

            candidate_neighbors = set()

            for v in c1:

                candidate_neighbors.update(vid_cid[v])

            candidate_neighbors.remove(i)

            for j in candidate_neighbors:

                c2 = cliques[j]

                if len(c2) < self._k:

                    continue

                if j < i:

                    continue

                s2 = set(c2)

                if len(s1 & s2) >= min(len(s1),len(s2)) -1:

                    clique_neighbor[i].add(j)

                    clique_neighbor[j].add(i) 
        # depth first search clique neighbors for communities

        communities = []

        for i,c in enumerate(cliques):

            if i in remained and len(c) >= self._k:

                #print 'remained cliques', len(remained)

                communities.append(set(c))

                neighbors = list(clique_neighbor[i])

                while len(neighbors) != 0:

                    n = neighbors.pop()

                    if n in remained:

                        #if len(remained) % 100 == 0:

                            #print 'remained cliques', len(remained)

                        communities[len(communities)-1].update(cliques[n])

                        remained.remove(n)

                        for nn in clique_neighbor[n]:

                            if nn in remained:

                                neighbors.append(nn)

        return communities
    
algorithm = CPM(G, 4)
com2 =algorithm.execute()
com2 = tuple(com2)

color_map = ["" for x in range(len(G))]
color = 0
for s in com2:
    indices = [i for i, x in enumerate(G.nodes) if x in s]
    print(indices)
    for i in indices:
        color_map[i] = Config.colors[color]
    color += 1

for i in range(len(color_map)):
    if color_map[i]=="":
        color_map[i] = Config.colors[-1]

print(color_map)

nx.draw(G, node_color=color_map, with_labels=True)
plt.show()