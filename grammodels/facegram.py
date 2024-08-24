import numpy as np
from sortedcontainers import SortedList, SortedSet

def sorted_intersection_min(key, sortedcontainer):
    if len(key) == 0:
        raise ValueError("key must have at least one element")

    if isinstance(sortedcontainer[0], SortedList):
        tmp = set(sortedcontainer[key[0]]).intersection(*[set(sortedcontainer[k]) for k in key[1:]])
        if len(tmp) > 0:
            return min(tmp)
        else:
            return [None]
    if isinstance(sortedcontainer[0], SortedSet):
        tmp = sortedcontainer[key[0]].intersection(*[sortedcontainer[k] for k in key[1:]])
        if len(tmp) > 0:
            return tmp.pop(0)
        else:
            return [None]


def facegram_alg1(mgm_treegrams, num_taxa=None, container='list'):
    """Compute the join facegram of a set of treegrams,
    which are given as a dictionary with block: [birth, death].

    The Algorithm works by computing -for each block in any of the treegrams-
    its minimal birth and minimal death over all treegrams which contain that block.
    Then we need to find for each block a superset with minimal birth time; those come
    from treegrams which do not contain the block. 
    If the minimal birth time of the superset is smaller or equal to the birth time
    of the block it has lifetime 0, otherwise its death is the minimal value of the
    death time of the block and the birth time of the superset.
    To do the last check, we can either pick a datastructure which allows for easy
    checks for supersets or we just iterate through the list of all possible blocks.

    Args:
        mgm_treegrams (dict): _description_
        num_taxa (int): _description_
        container (str, optional): _description_. Defaults to 'list'.

    Raises:
        ValueError: _description_
    """
    if not isinstance(mgm_treegrams[0], dict):
        raise ValueError("Treegrams should be a dictionary")

    mergegram = dict([])
    for tree in mgm_treegrams:
        for key, (birth, death) in tree.items():
            if key in mergegram:
                mergegram[key] = [min(mergegram[key][0], birth),
                                  min(mergegram[key][1], death)]
            else:
                mergegram[key] = [birth, death]

    if container == 'list':
        ## ATTENTION: DIFFERENT SORTING!
        mergegram = dict(sorted(mergegram.items(), key=lambda x: (x[1][0], -len(key))))

        keys = list(mergegram.keys())
        delete_keys = []
        deleted = False
        for (key, (birth, death)) in mergegram.items():
            setkey = set(key)
            for key_sup in keys:
                # check if the key is a subset
                if len(key) < len(key_sup) and setkey.issubset(set(key_sup)):
                    if mergegram[key_sup][0] <= birth:
                        delete_keys.append(key)
                        deleted = True
                    elif mergegram[key_sup][0] < death:
                        mergegram[key][1] = mergegram[key_sup][0]
                    break
            if deleted:
                keys.remove(key)
                deleted = False
        for key in delete_keys:
            del mergegram[key]
        del delete_keys
    else:
        if num_taxa is None:
            num_taxa = max(len(key) for key in mergegram)
            num_taxa = [key for key in mergegram if len(key) == num_taxa]

        taxa_sortedcontainers = [SortedSet() for _ in range(num_taxa)]
        # ATTENTION: DIFFERENT SORTING!
        mergegram = dict(sorted(mergegram.items(), key=lambda x: -len(x[0])))

        # prev_birth = np.inf
        # numeration = 0

        # containerSet = {}

        delete_keys = []
        for i, (key, (birth, death)) in enumerate(mergegram.items()):
            if i == 0:
                # prev_birth = birth
                for k in key:
                    # taxa_sortedcontainers[k].add((birth, numeration))
                    taxa_sortedcontainers[k].add((birth, hash(key)))
                continue

            birth_superset = sorted_intersection_min(key, taxa_sortedcontainers)[0]
            if birth_superset <= birth:
                delete_keys.append(key)
                continue
            if birth_superset < death:
                mergegram[key][1] = birth_superset

            # # adding to the containers
            # if birth == prev_birth:
            #     numeration += 1
            # else:
            #     numeration = 0
            #     prev_birth = birth
            
            # containerSet[(birth, numeration)] = key
            for k in key:
                # We should be using numbering of the element for a given filtration
                # value, but we rather use hashes - we only need to have different
                # values in the second position (whatever they are)
                taxa_sortedcontainers[k].add((birth, hash(key)))
        for key in delete_keys:
            del mergegram[key]
        # del delete_keys
        del delete_keys

    # return the labelled mergegram
    return(mergegram)


def facegram_alg2(mgm_treegrams, treedistances):
    # Get the union of all keys
    if isinstance(mgm_treegrams[0], dict):
        all_keys = set().union(*mgm_treegrams)
    elif isinstance(mgm_treegrams[0], list):
        all_keys = {tuple(y) for tg in mgm_treegrams
                    for x in tg for y in x[1]}
    else:
        raise ValueError("Treegrams should be a dictionary"
                         " or a list of lists")

    if not np.all([np.shape(treedistances[0]) == np.shape(td) for td in treedistances]):
        raise ValueError("All tree distances should have the same shape")
    if not isinstance(treedistances, np.ndarray):
        treedistances = np.array(treedistances)

    all_cliques = set(list(range(np.shape(treedistances)[-1])))

    # Initialize a dictionary to store the smallest values
    # just copy one directly
    mergegram = {}

    # Iterate over the keys and exploit that the treedistances
    # are a 3d array since they share the same taxa set
    for key in all_keys:
        birth = np.min(np.max(treedistances[:, *np.ix_(key, key)],
                              axis=(1,2)))
        # birth = np.min([np.max(matrix[np.ix_(key, key)])
        #                 for matrix in treedistances],
        #                axis=0)

        if len(all_cliques) == len(key):
            death = np.inf
        else:
            # get the largest element in each column, 
            # then the minimum of that;
            # we end up with a list from which we pick the minimal
            # value again
            death = np.min(np.max(treedistances[:,
                *np.ix_(key, list(all_cliques.difference(set(key))))],
                axis=1))
            # death = np.min([np.min(
            #     np.max(matrix[np.ix_(key,
            #         list(all_cliques.difference(set(key))))],
            #         axis=0))
            #     for matrix in treedistances])

        # Store the smallest value in the result dictionary
        if birth < death:
            mergegram[key] = [birth, death]

    return(mergegram)