# Source: https://github.com/shiruipan/TriDNR
# Input files: TriDNR/data/M10/{adjedges.txt, docs.txt, labels.txt}

# Attributes are constructed from titles of the original papers using TF/IDF scores.
# Stopwords and words occurring < 10 times are discarded.
# Nodes with classes [0, 1, 6, 7] are removed because of very low edge density within the classes.

# Resulting graph
# ---------------
# num_nodes = 4268
# num_edges = 5346
# num_attributes = 605
# num_classes = 6

import gust
import numpy as np
import scipy.sparse as sp

# List of nodes
nodes = []
with open('labels.txt') as f:
    for line in f:
        node, _ = line.split(maxsplit=1)
        nodes.append(node)

nodes = sorted(nodes)  # nodes are sorted lexicographically
i2n = dict(enumerate(nodes))  # index to node mapping
n2i = {v: k for (k, v) in i2n.items()}  # node to index mapping
n_nodes = len(n2i)

# Get node labels
z = -1 * np.ones(n_nodes)
with open('labels.txt') as f:
    for line in f:
        node, label = line.split()
        z[n2i[node]] = int(label)

assert np.all(z != -1)  # every node is labeled

# Construct adjacency matrix
A = sp.dok_matrix((n_nodes, n_nodes), dtype=np.int8)
with open('adjedges.txt') as f:
    for line in f:
        l_split = line.split()
        if len(l_split) > 1:
            i, js = l_split[0], l_split[1:]
            for j in js:
                try:
                    A[n2i[i], n2i[j]] = 1
                except:
                    pass  # node j is not known (has no label or attributes)
A = A.tocsr()

# Construct attribute list
X = dict()
with open('docs.txt') as f:
    for line in f:
        node, text = line.split(maxsplit=1)
        X[n2i[node]] = text.strip()
X = np.array([X[k] for k in sorted(X.keys())])

# Remove self-loops
A.setdiag(0)
A.eliminate_zeros()

# Remove nodes with degree 0
not_isolated = ((A.sum(0).A1 + A.sum(1).A1) > 0)
# Remove nodes belonging to classes [0, 1, 6, 7]
# because they are very sparsely connected in the network
good_classes = np.logical_not(np.in1d(z, [0, 1, 6, 7]))
keep_mask = np.logical_and(not_isolated, good_classes)
# Titles < 20 characters are garbage
long_titles = [i for (i, txt) in enumerate(X) if len(txt) > 20]
good_titles = np.zeros(n_nodes)
good_titles[long_titles] = 1
keep_mask = np.logical_and(keep_mask, good_titles)
nodes_to_keep = np.where(keep_mask)[0]

A = A[keep_mask][:, keep_mask]
X = X[keep_mask]
z = z[keep_mask]
# convert i2n map
new_idx = 0
new_i2n = {}
for old_idx in nodes_to_keep:
    new_i2n[new_idx] = i2n[old_idx]
    new_idx += 1
i2n = new_i2n


# Rename class labels
remap_z = dict(zip(np.unique(z), range(len(np.unique(z)))))
z = np.vectorize(remap_z.get)(z)

# Vectorize attributes

# Note that since we use min_df=10 you will get different results
# if you perform vectorization before or after filtering nodes from the network (e.g. singletons)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10, stop_words='english')
X_tfidf = tfidf.fit_transform(X)
i2a = {v: k for (k, v) in tfidf.vocabulary_.items()}

G = gust.SparseGraph(A, attr_matrix=X_tfidf, labels=z, idx_to_node=i2n, idx_to_attr=i2a)
gust.io.save_to_npz('citeseer_m10', G)
