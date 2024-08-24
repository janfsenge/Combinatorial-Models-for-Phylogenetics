""" Methods for computing the mergegram of a cliquegram from a distance matrix.

TODO: Extend the methods to work on data where the taxa are added later than the first step!
"""

# %%
# import packages
import numpy as np
import bisect
import networkx as nx
from tqdm import tqdm

# %%
# helper functions

def check_for_simple_data(dist_mat, filtrations):
    assert np.all(np.diag(dist_mat) == 0), "Diagonal of distance matrix is not all zero"
    assert 0 in filtrations, "Filtration 0 not in filtration list"
    assert len(filtrations) > 1

def extend_to_alive_cliques(cliques, prev_cliques, new_edges,
                            cutoff_exhaustive=2,
                            alwaysexhaustivesearch=False):
    """Given newly born cliques and a list of all alive cliques
    up to the previous spot, update the list to give 
    all still alive ones and return those and the dead ones
    """
    del_cliques = []
    if not alwaysexhaustivesearch and len(new_edges) == 1 and len(new_edges) <= cutoff_exhaustive:
        for cliq in cliques:
            # since we only have one new edge, all the new maximal cliques
            # must contain said edge
            y = sorted(set(cliq).difference(set(new_edges[0])))
            del_cliques.extend([y, y.copy()])
            bisect.insort(del_cliques[-2], new_edges[0][0])
            bisect.insort(del_cliques[-1], new_edges[0][1])

        # remove duplicates
        del_cliques = {tuple(sorted(list(set(x)))) for x in del_cliques}
        # not all to be deleted cliques are in the previous set,
        # hence get them out
        del_cliques = {tuple(x) for x in del_cliques if tuple(x) in prev_cliques}


    elif not alwaysexhaustivesearch and len(new_edges) == 2 and len(new_edges) <= cutoff_exhaustive:
        for cliq in cliques:
            # if we have two edges, let us check if the second (as well as the first)
            # are in the new cliques. At least of them has to be.
            if len(new_edges) > 1 and new_edges[1][0] in cliq and new_edges[1][1] in cliq:
                if new_edges[0][0] in cliq and new_edges[0][1] in cliq:
                    y = sorted(set(cliq).difference(set(new_edges[0])).difference(set(new_edges[1])))
                    del_cliques.extend([y, y.copy(), y.copy(), y.copy()])
                    bisect.insort(del_cliques[-4], new_edges[0][0])
                    bisect.insort(del_cliques[-4], new_edges[1][0])
                    bisect.insort(del_cliques[-3], new_edges[0][0])
                    bisect.insort(del_cliques[-3], new_edges[1][1])
                    bisect.insort(del_cliques[-2], new_edges[0][1])
                    bisect.insort(del_cliques[-2], new_edges[1][0])
                    bisect.insort(del_cliques[-1], new_edges[0][1])
                    bisect.insort(del_cliques[-1], new_edges[1][1])
                else:
                    y = sorted(set(cliq).difference(set(new_edges[1])))
                    del_cliques.extend([y, y.copy()])
                    bisect.insort(del_cliques[-2], new_edges[1][0])
                    bisect.insort(del_cliques[-1], new_edges[1][1])
            else:
                    # due to the construction new_edges[1] needs to be in cliq
                    y = sorted(set(cliq).difference(set(new_edges[0])))
                    del_cliques.extend([y, y.copy()])
                    bisect.insort(del_cliques[-2], new_edges[0][0])
                    bisect.insort(del_cliques[-1], new_edges[0][1])

        # remove duplicates
        del_cliques = {tuple(sorted(list(set(x)))) for x in del_cliques}
        # not all to be deleted cliques are in the previous set,
        # hence get them out
        del_cliques = {tuple(x) for x in del_cliques if tuple(x) in prev_cliques}

    else:
        # exhaustive version
        # TODO this can be done better, by breaking the loop once we have found the
        # print(prev_cliques, 'cliques', cliques)
        del_cliques = set([x for x in prev_cliques
                           if any([set(x).issubset(set(cliq)) for cliq in cliques])])
        # print('...', del_cliques)

    # now we get the all alive cliques by adding the new ones to the
    # cleaned previous cliques set
    cliques = prev_cliques.difference(del_cliques).union(cliques)
    return(cliques, del_cliques)


def cliquegram_alg1(dist_mat, filtrations=None, tqdm_disable=True):
    """Short version of the below; in general the iterative
    version is more efficient.
    This version is clearer to troubleshoot.
    """
    if filtrations is None:
        filtrations = np.unique(dist_mat)

    graph = nx.Graph()
    mask_upper_triu = np.triu(np.ones(np.shape(dist_mat)), 1).astype(bool)
    graph.add_nodes_from(np.arange(np.shape(dist_mat)[0]))
    prev_cliques = set([])
    labelled_mergegram = {}

    for thresh in tqdm(filtrations, disable=tqdm_disable, total=len(filtrations)):
        new_edges = [list(x) for x in zip(*np.where((dist_mat <= thresh) & mask_upper_triu))]
        graph.add_edges_from(new_edges)
        cliques = {tuple(sorted(x)) for x in nx.find_cliques(graph)}
        for cl in cliques.difference(prev_cliques):
            labelled_mergegram[cl] = [thresh, np.inf]
        for cl in prev_cliques.difference(cliques):
            labelled_mergegram[cl][1] = thresh
        prev_cliques = cliques

    return(labelled_mergegram)
    


def cliquegram_alg1_combined(dist_mat,
                             filtrations=None,
                             cutoff_warmstart=-np.inf,
                             alwaysexhaustivesearch=False,
                             cutoff_exhaustive=2,
                             tqdm_disable=True):
    """Simple implementation of cliquegram algorithm 1 - with only cold start for finding the cliques"""
    if filtrations is None:
        filtrations = np.unique(dist_mat)
    check_for_simple_data(dist_mat, filtrations)

    mask_upper_triu = np.triu(np.ones(np.shape(dist_mat)), 1).astype(bool)

    # create the auxiliary graph for which we do the clique detection
    graph = nx.Graph()
    # for the simple case we add all the nodes in the beginning
    graph.add_nodes_from(range(len(dist_mat)))

    # set up the labelled mergegram and the previous cliques
    if np.any(dist_mat[mask_upper_triu] < filtrations[1]):
        new_edges = [list(x) for x in 
                     zip(*np.where((dist_mat < filtrations[1])
                                   & mask_upper_triu))]
        graph.add_edges_from(new_edges)
        prev_cliques = {tuple(sorted(x)) for x in list(nx.find_cliques(graph))}
    else:
        prev_cliques = {tuple([x]) for x in range(len(dist_mat))}
    labelled_mergegram = {tuple(x): [filtrations[0], np.inf] for x in prev_cliques}

    for threshi, thresh in tqdm(enumerate(filtrations[1:]),
                                disable=tqdm_disable,
                                total=len(filtrations)-1):
        # print(thresh)
        # sinced we start filtrations from the second entry,
        # threshi can be used directly to access to previous'
        # filtrations elements
        new_edges = [list(x) for x in
                     zip(*np.where((dist_mat <= thresh) 
                        & (dist_mat > filtrations[threshi])
                        & mask_upper_triu))]
        
        # if there are no new edges, we can skip this iteration
        if len(new_edges) == 0:
            continue
        # add the new edges to the graph
        graph.add_edges_from(new_edges)

        
        if len(new_edges) <= cutoff_warmstart:
            # here we only have the new cliques, we need to
            # extend it to a list of all still alive cliques
            cliques = {tuple(sorted(y)) for edge in new_edges
                           for y in nx.find_cliques(graph, nodes=edge)}
            
            # CHECK: no x in cliques should be contained in labelled_mergegram
            # print(np.all([x not in labelled_mergegram for x in cliques]))
            
            # first add the newly born ones
            labelled_mergegram.update({x: [thresh, np.inf] for x in cliques})

            # now we need to find the ones that are still alive
            # since we only have a maximum of two new edges, we 
            # can do it explicitly instead of searching through the
            prev_cliques, del_cliques = \
                extend_to_alive_cliques(cliques, prev_cliques, new_edges,
                                        alwaysexhaustivesearch=alwaysexhaustivesearch,
                                        cutoff_exhaustive=cutoff_exhaustive)
            for x in del_cliques:
                labelled_mergegram[x][1] = thresh
        else:
            # now do a cold start to find all new cliques
            cliques = {tuple(sorted(x)) for x in list(nx.find_cliques(graph))}

            # add the newly born cliques to the labelled mergegram
            labelled_mergegram.update({x: [thresh, np.inf]
                for x in cliques.difference(prev_cliques)})
            # add death time to the one vanishing
            for x in prev_cliques.difference(cliques):
                labelled_mergegram[x][1] = thresh
        
            # now set prev_cliques to the current cliques
            prev_cliques = cliques

        # print([x for x in labelled_mergegram if labelled_mergegram[x][0] == 4])
    # now make the labelled mergegram into a mergegam
    return(labelled_mergegram)

# %%
# Algorithm 2 for finding cliques

def cliquegram_alg2_combined(dist_mat,
                             filtrations=None,
                             cutoff_warmstart=-np.inf,
                             use_prev_cliques=True,
                             make_distmatfitfiltration=True,
                             tqdm_disable=True):
    """Simple implementation of cliquegram algorithm 2"""
    if filtrations is None:
        filtrations = np.unique(dist_mat)

    if make_distmatfitfiltration:
        dist_mat = filtrations[np.digitize(dist_mat, bins=filtrations, right=True)]

    mask_upper_triu = np.triu(np.ones(np.shape(dist_mat)), 1).astype(bool)
    all_cliques = set(list(range(np.shape(dist_mat)[0])))

    # create the auxiliary graph for which we do the clique detection
    graph = nx.Graph()

    # graph.add_nodes_from(np.where(np.diag(dist_mat) == filtrations[0])[0])
    # nodes_later = np.where(np.diag(dist_mat) > filtrations[0])[0]
    check_for_simple_data(dist_mat, filtrations)
    graph.add_nodes_from(range(len(dist_mat)))

    # set up the labelled mergegram and the previous cliques
    if np.any(dist_mat[mask_upper_triu] < filtrations[1]):
        new_edges = [list(x) for x in 
                     zip(*np.where((dist_mat < filtrations[1])
                                   & mask_upper_triu))]
        graph.add_edges_from(new_edges)
        prev_cliques = {tuple(sorted(x)) for x in list(nx.find_cliques(graph))}
    else:
        prev_cliques = {tuple([x]) for x in range(len(dist_mat))}
    
    prev_only_new_cliques = False
    # we can directly add the death times
    # mergegram needs to be a list, since we can have the same entries multiple times
    if len(prev_cliques) == 1:
        if tuple(sorted(all_cliques)) in prev_cliques:
            return([[filtrations[0], np.inf]])
    mergegram = [[filtrations[0],
                  np.min(np.max(dist_mat[np.ix_(x, list(all_cliques.difference(set(x))))], axis=0))]
                 for x in prev_cliques]

    for threshi, thresh in tqdm(enumerate(filtrations[1:]),
                                disable=tqdm_disable,
                                total=len(filtrations)-1):
        # print(thresh)
        # sinced we start filtrations from the second entry,
        # threshi can be used directly to access to previous'
        # filtrations elements
        new_edges = [list(x) for x in
                     zip(*np.where((dist_mat <= thresh)
                        & (dist_mat > filtrations[threshi])
                        & mask_upper_triu))]

        # if there are no new edges, we can skip this iteration
        if len(new_edges) == 0:
            continue
        # add the new edges to the graph
        graph.add_edges_from(new_edges)

        if len(new_edges) <= cutoff_warmstart:
            # here we only have the new cliques, we need to
            # extend it to a list of all still alive cliques
            cliques = {tuple(sorted(y)) for edge in new_edges
                       for y in nx.find_cliques(graph, nodes=edge)}
            
            # CHECK: no x in cliques should be contained in labelled_mergegram
            # print(np.all([x not in labelled_mergegram for x in cliques]))
            if len(cliques) == 1:
                if tuple(sorted(all_cliques)) in cliques:
                    mergegram.append([thresh, np.inf])
                    break

            # we can add the new ones and their death times
            mergegram.extend([[thresh,
                np.min(np.max(dist_mat[np.ix_(x, list(all_cliques.difference(set(x))))], axis=0))
                ] for x in cliques])

            if use_prev_cliques:
                prev_cliques = cliques
                prev_only_new_cliques = True

        else:
            # now do a cold start to find all new cliques
            cliques = set(tuple(sorted(x)) for x in list(nx.find_cliques(graph)))

            # check that it's not the full clique
            if len(cliques) == 1:
                if tuple(sorted(all_cliques)) in cliques:
                    mergegram.append([thresh, np.inf])
                    break

            # check if we already used to find all cliques before
            if use_prev_cliques:
                # even tough we only added new cliques in the step prior
                # the set difference can still speed it up
                prev_cliques = cliques.difference(prev_cliques)
                if not prev_only_new_cliques:
                    mergegram.extend([[thresh, 
                        np.min(np.max(dist_mat[np.ix_(x, list(all_cliques.difference(set(x))))], axis=0))
                        ] for x in prev_cliques])
                else:
                    # we have to filter out the cliques which have already been born before
                    mergegram.extend([[thresh,
                        np.min(np.max(dist_mat[np.ix_(x, list(all_cliques.difference(set(x))))], axis=0))]
                        for x in prev_cliques
                        if np.max(dist_mat[np.ix_(x, x)]) > filtrations[threshi]])
                
                prev_only_new_cliques = False
                prev_cliques = cliques
            else:
                # we have to filter out the cliques which have already been born before
                mergegram.extend([[thresh,
                    np.min(np.max(dist_mat[np.ix_(x, list(all_cliques.difference(set(x))))], axis=0))]
                    for x in cliques
                    if np.max(dist_mat[np.ix_(x, x)]) > filtrations[threshi]])
    return(mergegram)