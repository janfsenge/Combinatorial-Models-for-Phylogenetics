# Numpy change it's output behavior;
# now it prints the type like np.float(0.0)
import numpy as np
# np.set_printoptions(legacy="1.25")

import networkx as nx

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from tqdm import tqdm

from sortedcontainers import SortedDict

# %%
# Class for some parts of the critical simplex diagram

class CriticalSimplexDiagram():
    """This class assumes two things:
    1. The only maximal clique is the the full simplex, hence we do
       not need to compute the maximal cliques nor have a maximal clique
       array as in the original critical simplex diagram
    2. The critical cells are sorted 

    Critical simplex diagram has been introduced in 


    TODO:
    - make a more efficient version of the intersection and union
    """

    def __init__(self, taxa, dist_mat=None,
                 filtration=None, subsampling=None,
                 save_as_integer_filtration=False):
        # save_as_integer_filtration

        # check inputs
        if taxa is None and dist_mat is None:
            raise ValueError("Either taxa or dist_mat must be provided")
        if dist_mat is not None:
            dist_mat = np.array(dist_mat)
            if not isinstance(dist_mat, np.ndarray):
                raise ValueError("Distance matrix must be a numpy array")
            else:
                try:
                    dist_mat = np.array(dist_mat)
                except:
                    raise ValueError("Distance matrix must be convertible to a numpy array")
            if dist_mat.ndim != 2:
                raise ValueError("Distance matrix must be 2D")
            if np.shape(dist_mat)[0] != np.shape(dist_mat)[1]:
                raise ValueError("Distance matrix must be square")

        # initialize the vertex array as SortedDict objects
        if taxa is not None:
            self.vertex_arrays = [SortedDict() for _ in range(len(taxa))]
        else:
            self.vertex_arrays = [SortedDict() for _ in range(dist_mat.shape[0])]
        
        # set the internal variables
        self.dist_mat = dist_mat
        self.X = taxa
        self.Graph = None
        self.integer_filtration = save_as_integer_filtration

        # for the filtation, we either use the unique values 
        # in the distance matrix or the provided filtration.
        # For references in the vertex arrays, we use integers
        # from 0, ..., len(filtrations)-1 to refer to the
        # filtration values
        if filtration is None:
            self.filtration = np.unique(dist_mat)
        else:
            self.filtration = filtration
        # Furthermore, we allow subsampling of the filtration.
        # TODO It might be more efficient to compute the
        # filtration once, and implement a collapse function
        # to get the subsampled filtration (if we want to compare
        # them in a larger fashion)
        if subsampling is not None:
            if isinstance(subsampling, (int, np.int32, np.int64)):
                self.filtration = \
                    np.linspace(np.min(self.filtration),
                                np.max(self.filtration),
                                subsampling)
            else:
                self.filtration = subsampling

        self.filtration = np.sort(self.filtration)
        self.num_critical_cells = np.zeros(self.filtration.shape[0])
        # since we do not include the full complex in the representation, we add it here
        self.num_critical_cells[-1] = 1

        if self.integer_filtration:
            self.full_complex = (len(filtration) - 1, 0)
        else:
            self.full_complex = (np.max(filtration), 0)
        self.full_complex_sigma = self.X

        # iterators
        self.iterable_index = None
        self.iterable_position = None
        self.current_iterable = None
    
    def _check_if_star_vertex(self, node_key, varrstart):
        if node_key in self.vertex_arrays[varrstart]:
            if isinstance(self.vertex_arrays[varrstart][node_key], (int, np.int32, np.int64)):
                return False
            return True
        return False

    def add_critical_nodes(self, criticalcells,
                           filtration_value=None,
                           filtration_index=None):
        """Add the critical nodes into the csd. Be aware that
        the node labels range from 0 to |K|-1 (i.e. |X|-1).

        Assumptions: we do not add critical cells for the same
        filtration value twice.

        Parameters
        ----------
        criticalcells : _type_
            Assume that critical cells are 'fixed' sorted
        filt_value : _type_
            filtration value
        """
        if filtration_value is None and filtration_index is None:
            raise ValueError("Either filtration_value or filtration_index must be provided")
        if filtration_value is not None:
            if filtration_value not in self.filtration:
                # self.filtration is sorted
                index = np.searchsorted(self.filtration, filtration_value)
                # add to the arrays
                self.filtration = np.insert(self.filtration, index, filtration_value)
                self.num_critical_cells = np.insert(self.num_critical_cells, index, 0)

                # update the filtration indices, when we use the integer filtration
                if self.integer_filtration:
                    self.full_complex = (self.full_complex[0] + 1, self.full_complex[1])

        if filtration_index is not None:
            filtration_value = self.filtration[filtration_index]

        index = np.searchsorted(self.filtration, filtration_value)
        if self.num_critical_cells[index] > 0:
            raise ValueError("Critical cells already added for this filtration value")
        
        if self.integer_filtration:
            filtration_value = index

        # now add the critical cells
        for Hsigma, vertex in enumerate(criticalcells):
            for i, v in enumerate(vertex):
                if i == 0:
                    # this might be an empty array
                    if len(vertex) > 1:
                        # vertex_arrays[v][(filtration[h], Hsigma)] = vertex[1:]
                        self.vertex_arrays[v].update({(filtration_value, Hsigma): vertex[1:]})
                    else:
                        # vertex_arrays[v][(filtration[h], Hsigma)] = None
                        self.vertex_arrays[v].update({(filtration_value, Hsigma): None})
                else:
                    # vertex_arrays[v][(filtration[h], Hsigma)] = vertex[0]
                    self.vertex_arrays[v].update({(filtration_value, Hsigma): vertex[0]})
        
        self.num_critical_cells[index] = Hsigma + 1

    def __iter__(self):
        self.iterable_index = np.argmax(self.num_critical_cells > 0)
        self.iterable_position = -1 # since we are gonna increase it in __next__

        if self.vertex_arrays:
            self.current_iterable = tuple([self.filtration[self.iterable_index],
                                           self.iterable_position])
        return self
    
    def __next__(self):
        # print(self.iterable_index, self.iterable_position)
        if self.current_iterable is None:
            raise ValueError("No iterable set")

        # we either have to increase the iterable index or the iterable position
        if self.iterable_position < self.num_critical_cells[self.iterable_index] - 1:
            self.iterable_position += 1
        else:
            # check if the are already at the end (which is at the last index)
            # since we previously checked that we are already at the last element
            # for this filtration value
            if (self.iterable_index == len(self.filtration) - 1 and
                self.iterable_position >= self.num_critical_cells[self.iterable_index] - 1):
                raise StopIteration
            
            # jump to the next filtration value we have critical cells;
            available_critical_cells = self.num_critical_cells[self.iterable_index+1:] > 0
            # # we could just exploit the face that we always have one critical cell
            # # at the end - the full one. So we don't need this check
            # if np.all(~available_critical_cells):
            #     raise StopIteration
            
            # this means that we have a filtration value with critical cells left
            self.iterable_index += 1 + np.argmax(available_critical_cells)
            self.iterable_position = 0
        
        # print('_', self.iterable_index, self.iterable_position)
        if self.integer_filtration:
            self.current_iterable = tuple([self.iterable_index,
                                           self.iterable_position])
        else:
            self.current_iterable = tuple([self.filtration[self.iterable_index],
                                        self.iterable_position])
        return(tuple(self.current_iterable))


    def update_values(self, dist_mat=None, filtration=None, subsampling=None):
        if dist_mat is not None:
            self.dist_mat = dist_mat
        if filtration is not None:
            self.filtration = filtration
        if subsampling is not None:
            if isinstance(subsampling, (int, np.int32, np.int64)):
                self.filtration = \
                    np.linspace(np.min(self.filtration),
                                np.max(self.filtration),
                                subsampling)
            else:
                self.filtration = subsampling
        return self

    
    # def add_critical_nodes(self, criticalcells,
    #                        filtration_value=None,
    #                        filtration_index=None):
    #     """add_critical_nodes _summary_

    #     Parameters
    #     ----------
    #     criticalcells : _type_
    #         Assume that critical cells are 'fixed' sorted
    #     filt_value : _type_
    #         filtration value
    #     """
    #     if filtration_value is None and filtration_index is None:
    #         raise ValueError("Either filtration_value or filtration_index must be provided")
    #     if filtration_index is not None:
    #         filtration_value = self.filtration[filtration_index]
        
    #     for Hsigma, vertex in enumerate(criticalcells):
    #         for i, v in enumerate(vertex):
    #             if i == 0:
    #                 # this might be an empty array
    #                 if len(vertex) > 1:
    #                     # vertex_arrays[v][(filtration[h], Hsigma)] = vertex[1:]
    #                     self.vertex_arrays[v].update({(filtration_value, Hsigma): vertex[1:]})
    #                 else:
    #                     # vertex_arrays[v][(filtration[h], Hsigma)] = None
    #                     self.vertex_arrays[v].update({(filtration_value, Hsigma): None})
    #             else:
    #                 # vertex_arrays[v][(filtration[h], Hsigma)] = vertex[0]
    #                 self.vertex_arrays[v].update({(filtration_value, Hsigma): vertex[0]})


    # def _check_if_star_vertex(self, node_key, varrstart):
    #     if node_key in self.vertex_arrays[varrstart]:
    #         if isinstance(self.vertex_arrays[varrstart][node_key], (int, np.int32, np.int64)):
    #             return False
    #         return True
    #     return False


    def _get_first_vertex_arrposition(self, node_key, varr_i):
        tmp_val = self.vertex_arrays[varr_i][node_key]

        if tmp_val is None:
            return varr_i
        if isinstance(tmp_val, (int, np.int32, np.int64)):
            return tmp_val
        return varr_i
    

    def _get_sigma(self, node_key, varrstart=None, varr_i=None):
        """_get_sigma _summary_

        Parameters
        ----------
        node_key : _type_
            _description_
        varrstart : _type_, optional
            varrstart is provided, return sigma if has its first vertex in there,
            by default None
        varr_i : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if self.integer_filtration and node_key[0] == len(self.filtration) - 1:
            return(self.full_complex_sigma)
        elif not self.integer_filtration and node_key[0] == self.filtration[-1]:
            return(self.full_complex_sigma)

        if varrstart is None:
            if varr_i is None:
                for vi in range(len(self.vertex_arrays)):
                    if node_key in self.vertex_arrays[vi]:
                        varrstart = vi
                        break
                if varrstart is None:
                    raise ValueError("node_key not found in any vertex array")

            else:
                varrstart = self._get_first_vertex_arrposition(node_key, varr_i)
        
        if node_key not in self.vertex_arrays[varrstart]:
            return []
        if isinstance(self.vertex_arrays[varrstart][node_key], (int, np.int32, np.int64)):
            return []

        sigma = [varrstart]
        tmp_val = self.vertex_arrays[varrstart][node_key]
        if tmp_val is not None:
            sigma += tmp_val

        return sigma


    def sorteddict_intersection(self, sigma, strategy=0):
        """Compute the intersection of the arrays in vertex_arrays given by sigma,
        it seems that strategy 0 is the fastest, not sure about large differences
        in the vertex arrays sizes.

        exclude sigma from the intersection
        """
        
        if len(sigma) == 0:
            return None

        if strategy == 0:
            intersection = self.vertex_arrays[sigma[0]].keys()

            # this will be the intersection of the keys
            for i in sigma[1:]:
                intersection &= self.vertex_arrays[i].keys()

        elif strategy == 1:
            idx = np.argmin([len(self.vertex_arrays[i])
                            for i in sigma])
            intersection = self.vertex_arrays[sigma[idx]].keys()

            for i in sigma[:idx] + sigma[idx+1:]:
                intersection &= self.vertex_arrays[i].keys()

        elif strategy == 2:
            idxs = np.argsort([len(self.vertex_arrays[i])
                            for i in sigma])
            idxs = np.array(sigma)[idxs]
            intersection = self.vertex_arrays[idxs[0]].keys()

            for i in idxs[1:]:
                intersection &= self.vertex_arrays[i].keys()

        else:
            return None

        if len(intersection) == 0:
            return None
    
        # now check if sigma is in there, otherwise just
        # return what's in the intersection
        tau_key = intersection[0]
        tau = self._get_sigma(tau_key, varr_i=sigma[0])
        if tau == sigma:
            return intersection[1:]
        else:
            return intersection


    def sorteddict_union(self, sigma):
        """Compute the intersection of the arrays in vertex_arrays given by sigma,
        it seems that strategy 0 is the fastest, not sure about large differences
        in the vertex arrays sizes.

        exclude sigma from the intersection
        """
        
        if len(sigma) == 0:
            return None

        union = self.vertex_arrays[sigma[0]].keys()

        # this will be the intersection of the keys
        for i in sigma[1:]:
            union |= self.vertex_arrays[i].keys()

        if len(union) == 0:
            return None
    
        return union


    # compute mergegram
    def compute_mergegram(self, labelled=False):
        """compute_mergegram _summary_

        Parameters
        ----------
        _param_ : _type_
            _description_
        """
        total = int(np.sum(self.num_critical_cells))
        for i, x in tqdm(enumerate(self), total=total):
            if i == total - 1:
                break
            clique = set(self._get_sigma(x))
            tmp = np.min(np.max(dist_mat[np.ix_(list(clique), list(taxa.difference(clique)))], axis=0))
            mgm_csd.append([x[0], tmp])

        mgm_csd = np.array(mgm_csd)
        mgm_csd = mgm_csd[np.lexsort([mgm_csd[:, 1], mgm_csd[:, 0]]), :]
        # mgm_csd = mgm_csd[mgm_csd[:, 1] != np.inf, :]
        return mgm_csd


    def compute_edges(self):
        self.Graph = nx.DiGraph()

        # add nodes
        for vi in range(len(self.vertex_arrays)):
            for node_key in self.vertex_arrays[vi]:
                if not self._check_if_star_vertex(node_key, vi):
                    continue
                sigma = self._get_sigma(node_key, varrstart=vi)
                self.Graph.add_node(node_key,
                                    label=sigma,
                                    birth=node_key[0],
                                    size=len(sigma))
        # add the full complex
        self.Graph.add_node(self.full_complex,
                            label=self.full_complex_sigma,
                            birth=self.full_complex[0],
                            size=len(self.full_complex_sigma))

        # now get the direct cofaces
        for vi in range(len(self.vertex_arrays)):
            for node_key in self.vertex_arrays[vi]:
                if not self._check_if_star_vertex(node_key, vi):
                    continue
                
                sigma = self._get_sigma(node_key, varrstart=vi)
                tau_candidates = self.sorteddict_intersection(sigma)

                tau_first = None
                while len(tau_candidates) > 0:
                    tau = tau_candidates.pop(0)
                    if tau_first is None:
                        tau_first = tau
                    elif tau == tau_first:
                        tau_candidates.append(tau)
                        break

                    remainder = self._get_sigma(tau, varr_i=vi)
                    remainder = [x for x in remainder if x not in sigma]
                    tau_cofaces = self.sorteddict_intersection(remainder)

                    if len(remainder) == 0:
                        assert tau_cofaces is None
                    if tau_cofaces is None:
                        assert len(remainder) == 0

                    if tau_cofaces is not None:
                        if len(tau_candidates) > 0 and len(tau_cofaces) > 0:
                            tau_candidates = [x for x in tau_candidates if x not in tau_cofaces]
                    tau_candidates.append(tau)
                
                if len(tau_candidates) > 0:
                    self.Graph.add_edges_from(
                        [[node_key, x] for x in tau_candidates]
                    )
                else:
                    self.Graph.add_edge(node_key, self.full_complex)
        
        return self.Graph


    def plot_graph(self, **kwargs):
        print(type(kwargs), isinstance(kwargs, dict))
        if self.Graph is None:
            self.Graph = self.compute_edges()

        relabel = {node: i for i, node in enumerate(self.Graph.nodes)}
        reverse = {i: node for node, i in relabel.items()}
        H = nx.relabel_nodes(self.Graph, relabel, copy=True)
        pos = nx.nx_pydot.graphviz_layout(H, prog="dot")
        pos = {reverse[key]: value for key, value in pos.items()}
        # now forcefully change the labels and put the nodes to their y-levels
        pos = {key: (pos[key][0], self.Graph.nodes('birth')[key]) for key, value in pos.items()}

        # set the parameters for the plot and check if they might already be in kwargs
        with_labels = True,
        labels = {node: self.Graph.nodes('label')[node] for node in self.Graph.nodes}
        if 'pos' in kwargs:
            pos = kwargs.pop('pos')
        if 'with_labels' in kwargs:
            with_labels = kwargs.pop('with_labels')
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
            
        nx.draw(self.Graph, pos=pos, with_labels=with_labels, labels=labels, **kwargs)



# %%
# MAIN

if __name__ == "__main__":
    X = np.arange(6)
    # maximal cliques
    M = [[[0], [1], [2], [3], [4], [5]],
        [[4,5]],
        [[2,4,5], [0,3]],
        [[0, 2, 3], [1,3]],
        [[0,1,2,3]]]
    filtration = [0, 1, 2, 3, 5, 7.5]

    csd = CriticalSimplexDiagram(taxa=X, filtration=filtration)
    for cells, filtval in zip(M, filtration[:len(M)]):
        csd.add_critical_nodes(criticalcells=cells, filtration_value=filtval)

    print(csd.compute_mergegram(labelled=False))
    csd.plot_graph()


# %%
# now for an more elaborate example

if __name__ == "__main__":
    X = np.ra
