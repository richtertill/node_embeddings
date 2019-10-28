# Source: https://github.com/shiruipan/TriDNR
# Input files: TriDNR/data/dblp/{adjedges.txt, docs.txt, labels.txt}

# Attributes are constructed from titles of the original papers using TF/IDF scores.
# Stopwords and words occurring < 10 times are discarded.
# You can change the way the attributes are constructed by uncommenting the lines below
attr_type = 'tfidf'
# attr_type = 'count'
# attr_type = 'binary'

# Resulting graph
# ---------------
# num_nodes = 17716
# num_edges = 105734
# num_attributes = 1639
# num_classes = 4

import gust
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

A = A[not_isolated][:, not_isolated]
X = X[not_isolated]
z = z[not_isolated]
nodes_to_keep = np.where(not_isolated)[0]
new_idx = 0
new_i2n = {}
for old_idx in nodes_to_keep:
    new_i2n[new_idx] = i2n[old_idx]
    new_idx += 1
i2n = new_i2n

# Vectorize features
if attr_type == 'tfidf':
    vectorizer = TfidfVectorizer(min_df=10, stop_words='english')
elif attr_type == 'binary':
    vectorizer = CountVectorizer(min_df=10, stop_words='english', binary=True)
elif attr_type == 'count':
    vectorizer = CountVectorizer(min_df=10, stop_words='english', binary=False)
else:
    raise ValueError("Unknown attr_type.")

X_vec = vectorizer.fit_transform(X)
i2a = {v: k for (k, v) in vectorizer.vocabulary_.items()}
i2c = {0: 'Databases', 1: 'Artificial_Intelligence', 2: 'Computer_Vision', 3: 'Data_Mining'}

G = gust.SparseGraph(A, attr_matrix=X_vec, labels=z,
                     idx_to_node=i2n, idx_to_attr=i2a, idx_to_class=i2c)
gust.io.save_to_npz('dblp', G)
