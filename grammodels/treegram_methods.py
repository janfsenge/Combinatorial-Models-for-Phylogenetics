"""
Collection of method for treegram handling and conversion 
between different representations.
"""

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# %% 
# treegram subroutines

def linkage2treegram(linkage, dist_mat):
    """Use a scipy linkage matrix to get the treegram list.
    It is a treegram list, since we use the values
    in the dist_mat to determine when singeltons
    are born.
    """
    labelled_mergegram = {}
    linkage_dict = {}
    treedistance = np.inf*np.ones(np.shape(dist_mat))
    np.fill_diagonal(treedistance, 0)

    for i, (x, y, dist, n) in enumerate(linkage):
        name = []
        if x < np.shape(dist_mat)[0]:
            birthdeath = [dist_mat[int(x), int(x)], dist]

            if birthdeath[0] != birthdeath[1]:
                labelled_mergegram[tuple([int(x)])] = birthdeath
        else:
            # the number must be in the linkage_dict
            key = linkage_dict[int(x) - np.shape(dist_mat)[0]]
            name = [tuple(key)]

            birthdeath = [linkage[int(x) - np.shape(dist_mat)[0], 2], dist]
            if birthdeath[0] != birthdeath[1]:
                labelled_mergegram[name[0]] = birthdeath

        if y < np.shape(dist_mat)[0]:
            birthdeath = [dist_mat[int(y), int(y)], dist]

            if birthdeath[0] != birthdeath[1]:
                labelled_mergegram[tuple([int(y)])] = birthdeath
        else:
            key = linkage_dict[int(y) - np.shape(dist_mat)[0]]
            name.append(tuple(key))
            treedistance[np.ix_(list(key), list(key))] = \
                np.min([treedistance[np.ix_(list(key), list(key))],
                    dist*np.ones_like(treedistance[np.ix_(list(key), list(key))])],
                    axis=0)
            
            birthdeath = [linkage[int(y) - np.shape(dist_mat)[0], 2], dist]
            if birthdeath[0] != birthdeath[1]:
                labelled_mergegram[name[-1]] = birthdeath
        
        # now add the new cluster to linkage_dict
        if len(name) == 0:
            key = sorted([int(x), int(y)])
        elif len(name) == 1:
            if x < np.shape(dist_mat)[0]:
                key = sorted(set(linkage_dict[int(y) - np.shape(dist_mat)[0]]).\
                             union(set([int(x)])))
            else:
                key = sorted(set(linkage_dict[int(x) - np.shape(dist_mat)[0]]).\
                             union(set([int(y)])))
        else:
            key = sorted(set(linkage_dict[int(x) - np.shape(dist_mat)[0]]).\
                         union(set(linkage_dict[int(y) - np.shape(dist_mat)[0]])))

        linkage_dict[i] = tuple(key)
        treedistance[np.ix_(list(key), list(key))] = \
                np.min([treedistance[np.ix_(list(key), list(key))],
                    dist*np.ones_like(treedistance[np.ix_(list(key), list(key))])],
                    axis=0)

        # print(linkage_dict)
        assert len(linkage_dict[i]) == n, f'{linkage_dict[i]} - {n}'
        if i == len(linkage) -1:
            labelled_mergegram[linkage_dict[i]] = [dist, np.inf]
            treedistance[treedistance == np.inf] = dist

    labelled_mergegram = dict(sorted(labelled_mergegram.items(), key=lambda x: (x[1][0], x[1][1]-x[1][0])))
    del linkage_dict
    return(labelled_mergegram, treedistance)


def treegram2ultramatrix(treegramlist,
                        dist_mat=None,
                        num_taxa=None,
                        check_is_tree=False):
    """Given a treegram list (i.e. a list of [threshold, list of blocks for this threshold])
    convert that into it's ultramatrix representation.
    """

    if num_taxa is None:
        if isinstance(treegramlist, dict):
            num_taxa = len(set([]).union(*treegramlist))
        elif isinstance(treegramlist, list):
            num_taxa = len(set([tuple(y) for x in treegramlist for y in x[1]]))

    ultramatrix = np.inf * np.ones([num_taxa, num_taxa])
    if isinstance(treegramlist, dict):
        for block, block_birthdeath in treegramlist.items():
            submatrix = ultramatrix[np.ix_(block, block)]
            ultramatrix[np.ix_(block, block)] = \
                np.min([submatrix, block_birthdeath[0]*np.ones_like(submatrix)], axis=0)
    elif isinstance(treegramlist, list):
        for block_birthdeath in treegramlist:
            for block in block_birthdeath[1]:
                submatrix = ultramatrix[np.ix_(block, block)]
                ultramatrix[np.ix_(block, block)] = \
                    np.min([submatrix, block_birthdeath[0]*np.ones_like(submatrix)], axis=0)

    if check_is_tree:
        if dist_mat is None:
            raise ValueError('dist_mat must be provided if check_tree is True')
        if np.all(ultramatrix == dist_mat):
            return(ultramatrix, True)
        return(ultramatrix, False)
    return(ultramatrix)


def ultramatrix2treegram(treedistance, 
        check_is_tree=False,
        check_tolerance=1e-14):
    if np.shape(treedistance)[0] != np.shape(treedistance)[1]:
        raise ValueError('Distance matrix is not square!')

    num_taxa = np.shape(treedistance)[0]

    taxas_remaining = list(range(num_taxa))
    all_taxa = set(taxas_remaining)
    blocks_birthdeath = {tuple(taxas_remaining):
                         [np.max(treedistance), np.inf]}
    
    while len(taxas_remaining) > 0:
        tupleblock = tuple([taxas_remaining.pop()])
        current_birth = treedistance[tupleblock[0], tupleblock[0]]
        # directly add all the other taxa which have the
        # same
        tupleblock = tuple(np.where(treedistance[tupleblock[0], :]
                                    == current_birth)[0])

        other_taxas = all_taxa.difference(set(tupleblock))
        list_others = list(other_taxas)

        # now start the loop
        while tupleblock not in blocks_birthdeath and len(list_others) > 0:
            newblockmembers = np.min(treedistance[np.ix_(tupleblock,
                                list_others)],
                                axis=0)
            current_death = np.min(newblockmembers)
            blocks_birthdeath[tupleblock] = [current_birth, current_death]

            # check if it is really a tree distance
            if check_is_tree:
                if not np.all(np.std(treedistance[np.ix_(tupleblock, list_others)],
                                     axis=0) < check_tolerance):
                    raise ValueError('Distance matrix does not seem to be a tree distance matrix!')

            # update loop iterables
            idx = np.where(newblockmembers == current_death)[0]

            tupleblock = tuple(sorted(list(tupleblock) + [list_others[i] for i in idx]))
            current_birth = current_death

            other_taxas = other_taxas.difference(set(tupleblock))
            list_others = list(other_taxas)

    return blocks_birthdeath


def ultramatrix2treegramScipy(treedistance,
        method='single',
        lifetime_threshold=1e-15,
        check_is_tree=False,
        ignore_error=True):
    """Reconstruct a labelled mergegram from the treedistances given
    via scipy linkage.
    If the distances are truly tree distances, it does not matter which
    hierarchical clustering algorithm we use, since there should be no
    difference in the treegrams.
    BE AWARE: when using methods like 'average' then the 
    resulting birth and death values might be slighlty different,
    in particular there might be blocks with a lifetime very close
    to 0 which need to be filtered out.
    """
    # scipy only allows for 0 diagonals
    diagonals = np.diag(treedistance).copy()
    np.fill_diagonal(treedistance, 0)

    linkage = sch.linkage(squareform(treedistance), method=method)
    treegramdict, treedistance2 = linkage2treegram(linkage, treedistance)

    # filter out the blocks which might have been
    # created by inaccuracies during hierarchical clustering
    if method not in ['single', 'complete']:
        for key, value in treegramdict.items():
            if value[1] - value[0] < lifetime_threshold:
                del treegramdict[key]

    # now add back the singletons
    for singleton, thresh in enumerate(diagonals):
        key = tuple([singleton])
        if key in treegramdict:
            treegramdict[key][0] = thresh
            # if the birth and death are the same, we can remove it
            if treegramdict[key][0] == treegramdict[key][1]:
                del treegramdict[key]
        else:
            # this can only happen if the singleton has already been
            # killed by a superset at level 0
            # by definition thresh needs to be 0 as well, and we do not do anything
            if thresh != 0:
                raise ValueError(f'The singleton {singleton} has a higher value'
                                 ' in the diagional than elsewhere!')

    treegramdict = dict(sorted(treegramdict.items(),
                               key=lambda x: (x[1][0], x[1][1]-x[1][0])))

    if check_is_tree:
        if ignore_error:
            return(treegramdict, np.all(treedistance2 == treedistance))

        if np.any(treedistance2 != treedistance):
                raise ValueError('The distances are not treedidistances!')

    return(treegramdict)


def treegramlist2treegram(treegramlist):
    # check input
    if np.all([len(x) == 2 for x in treegramlist]):
        if not np.all([treegramlist[i][0] < treegramlist[i+1][0]
                       for i in range(len(treegramlist)-1)]):
            treegramlist = [treegramlist[i] for i in np.argsort([x[0] for x in treegramlist])]
    else:
        raise ValueError("Treegramlist should be a list of pairs")

    treegramlist = [(x[0], set([tuple(sorted(y)) for y in x[1]])) for x in treegramlist]
    treegramdict = {x: [treegramlist[0][0], np.inf] for x in treegramlist[0][1]}
    for i in range(1, len(treegramlist)):
        # new blocks
        for block in treegramlist[i][1].difference(treegramlist[i-1][1]):
            treegramdict[block] = [treegramlist[i][0], np.inf]
        # killed blocks
        for block in treegramlist[i-1][1].difference(treegramlist[i][1]):
            treegramdict[block][1] = treegramlist[i][0]
    return(treegramdict)