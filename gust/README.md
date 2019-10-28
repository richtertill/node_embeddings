# Graph Utilities and STorage (GUST)
This library includes methods for storage and preprocessing of graphs 
represented as Numpy arrays and sparse matrices, as well a collection of graph
datasets.
We decided to write our own library instead of using existing libraries like 
NetworkX since 
- many important graph algorithms are missing from NetworkX, or are very 
inefficiently implemented there;
- we usually represent graphs as matrices (adjacency matrix, attribute matrix, 
etc.), which is why it's more natural to also store them in this format 
(instead of converting each time);
- our approach is much more efficient in terms of storage space and I/O speed.


## Contributing
If you would like to add some code to the library, please make sure that you
follow the guidelines described in [CONTRIBUTING.md](CONTRIBUTING.md).
