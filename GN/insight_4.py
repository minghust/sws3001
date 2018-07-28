import networkx as nx
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image
import numpy as np
import itertools
import math
import os

"""## Helper functions"""

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

"""---


## Build graph
Construct an adjacency matrix from the dissimilarity matrix, then use the adjacency matrix to build a networkx graph
"""

# Load a saved copy of the dissimilarity matrix

simfilename = "insight_4_sim.tsv"
labelfilename = "insight_4_label.csv"
sum_labels=0
X = np.genfromtxt(simfilename, delimiter=' ', encoding='utf8', dtype=None)
# print(X)
label2idx = dict()
sim = np.zeros((500, 500))
sim.fill(100)
if os.path.exists(labelfilename):
    os.remove(labelfilename)
f = open(labelfilename, 'w')
for t in X:
    t0, t1, t2 = t
    if not t0 in label2idx:
        idx = len(label2idx)
        label2idx[t0] = idx
        f.write(str(idx) + "," + t0 + "\n")
        sum_labels += 1
    if not t1 in label2idx:
        idx = len(label2idx)
        label2idx[t1] = len(label2idx)
        f.write(str(idx) + "," + t1 + "\n")
        sum_labels += 1
    idx0 = label2idx[t0]
    idx1 = label2idx[t1]
    sim[idx0][idx1] = t2
    sim[idx1][idx0] = t2


f.close()

Config.labels = []
with open(labelfilename) as f:
    for line in f:
        _id, label = line.rstrip().split(",")
        _type = str(int((int(_id)/20)))+'-'+str(int(_id)%20)
        label.split("__")[0]
        Config.labels.append(label.split("__")[0]+'-'+str(int(_id)%20))

# print("Loaded labels (" + str(len(Config.labels)) + " classes): ", end='')
# print(Config.labels)

# Analyze distribution of dissimilarity score
simflat = sim.copy()
simflat = simflat[simflat != 100] # Too many 100 result in a bad histogram so we remove them
# _ = plt.hist(simflat, bins=25)

mmax  = np.max(simflat)
mmin  = np.min(simflat)
mmean = np.mean(simflat)
print('avg={0} min={1} max={2}'.format(mmean, mmin, mmax))

# Select a suitable threshold and set dissimilarity scores larger than that threshold to zero

threshold = 0.8
adjmat = sim.copy()
np.fill_diagonal(adjmat, np.min(simflat)) # Set the diagonal elements to a small value so that they won't be zeroed out

for x in np.nditer(adjmat, op_flags=['readwrite']):
    if x <= threshold:
        x[...] = 1
    else:
        x[...] = 0

np.fill_diagonal(adjmat, np.min(simflat)) # Set the diagonal elements to a small value so that they won't be zeroed out

print(sum_labels)
adjmat = adjmat[0:sum_labels, 0:sum_labels]
print(len(adjmat))
# adjmat = adjmat.reshape(sim.shape)

np.set_printoptions(threshold=np.nan)
# print(adjmat)

# Construct a networkx graph from the adjacency matrix
# (Singleton nodes are excluded from the graph)
G = make_graph(adjmat, labels=Config.labels)
# nx.draw(G, with_labels=True)

"""---


# ##Community detection using Girvan-Newman
# """

from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import modularity
# from networkx.algorithms.community import k_clique_communities

comp = girvan_newman(G)

max_shown = 1
shown_count = 1
possibilities = []
for communities in itertools.islice(comp, max_shown):
    # print(communities) # a tuple
    # print("Possibility", shown_count, ": ", end='')
    # print(communities)
    print('modularity',modularity(G, communities))
    possibilities.append(communities)
    color_map = ["" for x in range(len(G))]
    color = 0
    for c in communities:
        indices = [i for i, x in enumerate(G.nodes) if x in c]
        for i in indices:
            color_map[i] = Config.colors[color]
        color += 1
    print('styles: ',color)
    shown_count += 1
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

# Generate and download edges for Gephi

export_edge_list(adjmat, labels=Config.labels, filename='gephi-edges.csv')

###Q1. Generate the edge file and community file for the pals system.
# Generate the community file for pals system

which_possibility = 1

communities = possibilities[which_possibility-1]

indices_in_community = []   # For obtaining submatrix of adjmat

f = open("pals-insight4-community.dat", 'w')
cur_com = 1
for c in communities:
    indices = [i for i, x in enumerate(Config.labels) if x in c]
    indices_in_community.extend(indices)
    for i in indices:
         f.write("\"" + Config.labels[i] + "\" " + str(cur_com) + "\r\n")
    cur_com += 1
f.close()


# Generate the graph file for pals system

# Obtain the submatrix of adjmat with only elements that appear in communities
indices_in_community = sorted(indices_in_community)
adjmat_in_community = adjmat[indices_in_community,:][:,indices_in_community]

# Obtain sublist of labels of only elements that appear in communities
labels = np.array(Config.labels)[indices_in_community].tolist()

export_edge_list(adjmat_in_community, labels=labels, filename='pals-insight4-edges.dat', delim=" ", header=False)

"""###Q2. Find the node with the most number of neighbors"""

# add each node with its number of neighbors as a sub-list to a list called `label_neighbors`
label_neighbors = [] # record the label and its number of neighbors
label_index=0
for each_label in Config.labels:
    count = 0
    for dis in adjmat[label_index]:
        if dis != 0:
            count += 1
    label_neighbors.append(list((each_label, count)))
    label_index += 1

# print(label_neighbors) # it should be [['AisazuNihaIrarenai', 6], ['AkkeraKanjinchou', 16],...] , means that AisazuNihaIrarenai has 6 neighbors, and AkkeraKanjinchou has 16 neighbors, and so on.

# find the node with the most number of neighbors
node=[]
max_neighbors = label_neighbors[0][1]
for each_pair in label_neighbors:
    if each_pair[1] > max_neighbors:
        max_neighbors = each_pair[1]
        node = each_pair

# print result
# print('The node is',node[0],',and it has',node[1],'neighbors.\nIt has the most number of neighbors.')

"""### Q3. List the neighbors of the node (with the most neighbors) found above"""

# get the index of the node's index
label_index = label_neighbors.index(node)

# print('Print the neighbors of the node(EXCEPT the node itself), and the neighbor\'s index.\n')
neighbor_index=0
for distance in adjmat[label_index]:
    if distance!=0 and Config.labels[neighbor_index]!=node[0]:
        pass
        # print('Neighbor:',Config.labels[neighbor_index],', Index:', neighbor_index)
    neighbor_index += 1

"""###Q4. Plot a histogram of the degrees of the nodes in the graph"""

# the degree of each node means the neighbors of each node
# firstly, get the x, y
x = list(range(0, len(Config.labels))) # each label's index
y = [label_neighbors[i][1] for i in x] # each label's number of neighbors

# secondly, set graph's title and x,y label
plt.title('Histogram of the degrees of all nodes', fontsize=20)
plt.xlabel('node index', fontsize=15)
plt.ylabel('number of neighbors', fontsize=15)
# finally, plot and show
plt.bar(x, y)
# plt.show()

"""###Q5. Produce a list of the nodes sorted from that with the most neighbors to that with the least neighbors"""

# install the prettytable
from prettytable import PrettyTable 

sorted_label_neighbors = sorted(label_neighbors, key=lambda item: item[1], reverse=True)
table = PrettyTable(['Node', 'Number of Neighbors'])
for item in sorted_label_neighbors:
    table.add_row(item)
# print(table)
