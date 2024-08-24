# Combinatorial Models for Phylogenetics
 Code repository for the algorithms and code of the paper "Combinatorial Topological Models for Phylogenetic Networks and the Mergegram Invariant" by Paweł Dłotko, Jan F Senge and Anastasios Stefanou


The code consists of implementations of computing the mergegram invariant of different concepts introduced in the paper. We exploit their relationships as seen in the following image

![image](latex/concepts_overview_connections.pdf)

It should be noted, that given one treegram, the mergegram of the treegram is the same as the mergegram of the join-cliquegram of the treegram as well as the mergegram of the join-facegram of the treegram. 

In these computations, we don't save the intermediate results to save space and thus we do not construct the cliquegrams, facegrams explicitly. A naiive implementations would be lists of all maximal blocks (treegram), maximal cliques (cliquegram), maximal faces (facegram) together with the value at which they appear (filtration value). A more efficient implementation uses the data structure of critical simplex diagram (see [An Efficient Representation for Filtrations of Simplicial Complexes (Boissonnat, Karthik)](https://arxiv.org/pdf/1607.08449)).
