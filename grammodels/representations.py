"""
This module contains the meta-classes for Cliquegram and Facegram representations. 

The necessarry methods to compute the mergegram of the cliquegram (join and in general) as well 
as the facegram of the join are implemented. 

by Jan Felix Senge, 13.08.2024
"""

# %%
# import the necessary packages

# import inspect
import numpy as np
from sortedcontainers import SortedSet, SortedList
from tqdm import tqdm

from grammodels.cliquegram import cliquegram_alg1, cliquegram_alg1_combined, cliquegram_alg2_combined
from grammodels.facegram import facegram_alg1, facegram_alg2
import grammodels.treegram_methods as treegram_methods
# from grammodels.simplexdiagram import CriticalSimplexDiagram

# %%
#

def sorted_intersection_min(key, sortedcontainer, sigma_key=None):
    if len(key) == 0:
        raise ValueError("key must have at least one element")

    if isinstance(sortedcontainer[0], SortedList):
        tmp = set(sortedcontainer[key[0]]).intersection(*[set(sortedcontainer[k]) for k in key[1:]])
        if sigma_key is not None:
            tmp.discard(sigma_key)
        if len(tmp) > 0:
            return min(tmp)[0]
        else:
            return None
    if isinstance(sortedcontainer[0], SortedSet):
        tmp = sortedcontainer[key[0]].intersection(*[sortedcontainer[k] for k in key[1:]])
        if sigma_key is not None:
            tmp.discard(sigma_key)
        if len(tmp) > 0:
            return tmp.pop(0)[0]
        else:
            return None

def format_mergegram(mergegram, delete_infinite=False,
                     filterthreshold=0):
    if isinstance(mergegram, dict):
        mergegram = np.array(list(mergegram.values()))
    else:
        try:
            mergegram = np.array(mergegram)
        except Exception as exc:
            raise ValueError("Mergegram should be a dictionary or numpy convertible") from exc
    if mergegram.ndim != 2:
        raise ValueError("Mergegram should be a 2D array") 
    if delete_infinite:
        mergegram = mergegram[mergegram[:, 1] != np.inf]
    if filterthreshold > 0:
        mergegram = mergegram[mergegram[:, 1] - mergegram[:, 0] > filterthreshold]

    return(mergegram[np.lexsort([mergegram[:,1], mergegram[:,0],
                                 mergegram[:,1] - mergegram[:,0]])])


def compare_mergegrams(mergegram1,
                       mergegram2=None,
                       filterthreshold=1e-16,
                       tolerance=1e-15,
                       return_exact_differences=False):
    """Compare two mergegrams and return if they are the same. Also allows iterables of mergegrams.

    Args:
        mergegram1 (dict, list, iterable): _description_
        mergegram2 (_type_, optional): mergegram to compare to.
            If not given, mergegram1 needs to be an iterable. Defaults to None.
        return_exact_differences (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_

    Returns:
        bool: if all of the mergegrams are the same
        dict: i return_exact_differences
    """
    if mergegram2 is None:
        try:
            iterator = iter(mergegram1)
        except TypeError as exc:
            raise TypeError('If giving just mergegram1,'
                            'it should be an iterable of mergegrams') from exc

        mgm1 = next(iterator)
        if return_exact_differences and isinstance(mgm1, dict):
            diff_inmgm1 = []
            diff_notinmgm1 = []
            compare_bool = True

            for mgm2 in iterator:
                tmp = compare_mergegrams(mgm1, mgm2,
                                         return_exact_differences=True,
                                         filterthreshold=filterthreshold)
                compare_bool &= tmp[0]
                diff_inmgm1.append(tmp[1])
                diff_notinmgm1.append(tmp[2])

            return compare_bool, diff_inmgm1, diff_notinmgm1

        return np.all([compare_mergegrams(mgm1, mgm2)
                       for mgm2 in iterator])

    if return_exact_differences and isinstance(mergegram1, dict):
        diff1 = set(mergegram1.keys()).difference(set(mergegram2.keys()))
        diff2 = set(mergegram2.keys()).difference(set(mergegram1.keys()))

        x1 = format_mergegram(mergegram1, filterthreshold=filterthreshold)
        x2 = format_mergegram(mergegram2, filterthreshold=filterthreshold)
        if len(x1) != len(x2):
            x1sameasx2 = False
        else:
            x1sameasx2 = np.allclose(x1, x2, atol=tolerance, rtol=0)
        return x1sameasx2, diff1, diff2

    return np.allclose(format_mergegram(mergegram1, filterthreshold=filterthreshold),
            format_mergegram(mergegram2, filterthreshold=filterthreshold),
            atol=tolerance, rtol=0)


# %%
# treegrams clas

class Treegrams():
    """Class for handling a list of treegrams and convert between
    their different representations.

    The treegram is represented internally via a dicitionary,
    which is the same as the labelled mergegram.

    TODO: right now we rather focus on dendrograms, i.e. 
    the distance matrices have zeros in the diagoals
    """
    def __init__(self,
                 treedistances=None,
                 treegrams_representation=None,
                 linkages=None,
                 linkage_distances=None):

        if linkages is not None and linkage_distances is not None:
            self.treegramslist = \
                [treegram_methods.linkage2treegram(linkage,
                    linkage_distances) for linkage in linkages]
            # linkage2treegram returns treegram, treedistance
            self.treedistances = [x[1] for x in self.treegramslist]
            self.treegramslist = [x[0] for x in self.treegramslist]

        if treedistances is not None:
            self.treedistances = treedistances

        if treegrams_representation is not None:
            if np.all([isinstance(x, list) for x in treegrams_representation]):
                self.treegramslist = [treegram_methods.treegramlist2treegram(x)
                                  for x in treegrams_representation]
            elif np.all([isinstance(x, dict) for x in treegrams_representation]):
                self.treegramslist = treegrams_representation
            else:
                raise ValueError("Treegrams should be a list of dictionaries "
                    + "or a list of maximal blocks with filtration value")
            
            if self.treedistances is None:
                self.treedistances = \
                    np.array([treegram_methods.treegram2ultramatrix(x)
                              for x in self.treegramslist])

        self.num_taxa = np.shape(self.treedistances[0])[0]
        # check: all the same number of taxa?!

    def compute_labelled_mergegram(self,
                                   method='single',
                                   check_is_tree=False,
                                   lifetime_threshold=1e-15,
                                   check_tolerance=1e-14):
        if self.treedistances is None:
            raise ValueError("Treedistance need to be set!")

        if method is None:
            self.treegramslist = \
                [treegram_methods.ultramatrix2treegram(td,
                    check_is_tree=check_is_tree,
                    check_tolerance=check_tolerance)
                for td in self.treedistances]
        else:
            self.treegramslist = \
                [treegram_methods.ultramatrix2treegramScipy(td,
                    method,
                    lifetime_threshold=lifetime_threshold,
                    check_is_tree=check_is_tree,
                    ignore_error=False)
                for td in self.treedistances]
        
        return self.treegramslist

# %%
# do the cliquegram class

class Cliquegram():
    def __init__(self,
                 distancematrix=None,
                 algorithm='Alg1',
                 delete_infinite=False):

        self.algorithm = algorithm
        self.delete_infinite = delete_infinite

        self.distancematrix = distancematrix
        self.mergegram = None
    
    def minimizer_from_treegrams(self, treedistances, axis=0):
        if isinstance(treedistances, list):
            treedistances = np.array(treedistances)
        if treedistances.ndim == 2:
            if treedistances.shape[1] != treedistances.shape[0]:
                raise ValueError("If giving one treegram, it should be a square matrix")
        elif treedistances.ndim != 3:
            raise ValueError("Treegrams should be a 3D array or a 2D array with 3 columns")

        self.distancematrix = np.min(treedistances, axis=axis)
        return self
        # return self.distancematrix
        
    def compute_mergegram(self,
            distancematrix=None,
            algorithm=None,
            filtrations=None,
            cutoff_warmstart=2,
            alwaysexhaustivesearch=False,
            cutoff_exhaustive=2,
            use_prev_cliques=True,
            make_distmatfitfiltration=True,
            tqdm_disable=True):

        if distancematrix is not None:
            self.distancematrix = distancematrix
        if self.distancematrix is None:
            raise ValueError("Distance matrix or cliquegram need to be set!")

        if algorithm is not None:
            self.algorithm = algorithm
        # if self.algorithm == 'Alg1':
        #     target_method = cliquegram_alg1_combined
        # elif self.algorithm == 'Alg2':
        #     target_method = cliquegram_alg2_combined
        # else:
        #     raise ValueError("Algorithm not implemented")

        # # Get the signature of the target method
        # sig = inspect.signature(target_method)
        # # Filter kwargs to include only valid parameters for the target method
        # valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        # # Call the target method with the filtered kwargs
        # self.mergegram = target_method(self.distancematrix, **valid_kwargs)

        if self.algorithm == 'Alg1':
            self.mergegram = cliquegram_alg1(self.distancematrix,
                                filtrations=filtrations,
                                tqdm_disable=tqdm_disable)

        elif self.algorithm == 'Alg1combined':
            self.mergegram = cliquegram_alg1_combined(self.distancematrix,
                filtrations=filtrations,
                cutoff_warmstart=cutoff_warmstart,
                alwaysexhaustivesearch=alwaysexhaustivesearch,
                cutoff_exhaustive=cutoff_exhaustive,
                tqdm_disable=tqdm_disable)

            # make it so that it is sorted by the filtration value
            self.mergegram = dict(sorted(self.mergegram.items(), key=lambda x: (x[1][0], x[1][1]-x[1][0])))

        elif self.algorithm == 'Alg2':
            self.mergegram = cliquegram_alg2_combined(self.distancematrix,
                filtrations=filtrations,
                cutoff_warmstart=cutoff_warmstart,
                use_prev_cliques=use_prev_cliques,
                make_distmatfitfiltration=make_distmatfitfiltration,
                tqdm_disable=tqdm_disable)

        else:
            raise ValueError("Algorithm not implemented, pick Alg1, Alg1combined or Alg2")
        return self.mergegram

    def return_mergegram(self, delete_infinite=None):
        if delete_infinite is not None:
            self.delete_infinite = delete_infinite

        if self.mergegram is None:
            if self.distancematrix is not None:
                self.compute_mergegram()
            else:
                raise ValueError("Distance needs to be given")
    
        return format_mergegram(self.mergegram, self.delete_infinite)

# %%
# Do the facegram class

class Facegram():
    # TODO: add the possibility to get the mergegram for given list of maximal faces
    # TODO: add the integration to get mergegram from general filtration (gudhi?)
    def __init__(self,
                 facegram=None,
                 treegrams=None,
                 treegramslist=None,
                 treedistances=None,
                 algorithm='Alg2',
                 delete_infinite=False):

        self.algorithm = algorithm
        self.delete_infinite = delete_infinite

        self.facegram = facegram
        if isinstance(treegrams, Treegrams):
            self.treedistances = treegrams.treedistances
            if treegrams.treegramslist is None:
                self.treegramslist = treegrams.compute_labelled_mergegram()
            else:
                self.treegramslist = treegrams.treegramslist
        elif treedistances is not None:
            self.treedistances = treedistances
            if treegramslist is not None:
                self.treegramslist = treegramslist
            else:
                self.treegramslist = \
                    [treegram_methods.ultramatrix2treegram(td)
                     for td in self.treedistances]
        elif treegramslist is not None:
            self.treegramslist = treegramslist
            self.treedistances = \
                [treegram_methods.treegram2ultramatrix(tg)
                 for tg in treegramslist]

        self.mergegram = None

    def compute_join(self,
                     treegrams=None,
                     treegramslist=None,
                     treedistances=None,
                     algorithm=None,
                     alg1container='list'):

        # check the inputs
        if algorithm is not None:
            self.algorithm = algorithm

        if treegramslist is not None:
            self.treegramslist = treegramslist
            if treedistances is not None:
                self.treedistances = treedistances
            else:
                self.treedistances = \
                    [treegram_methods.treegram2ultramatrix(tg)
                     for tg in treegramslist]

        elif treedistances is not None:
            self.treedistances = treedistances
            if treegramslist is not None:
                self.treegramslist = treegramslist
            else:
                self.treegramslist = \
                    [treegram_methods.ultramatrix2treegram(td)
                     for td in self.treedistances]

        elif treegrams is not None:
            if not isinstance(treegrams, Treegrams):
                raise ValueError("If given Treegrams should be a Treegrams object")
            self.treedistances = treegrams.treedistances
            if treegrams.treegramslist is None:
                self.treegramslist = treegrams.compute_labelled_mergegram()
            else:
                self.treegramslist = treegrams.treegramslist

        # now check if all been set
        if self.treedistances is None:
            raise ValueError("Treedistances need to be given!")
        if self.treegramslist is None:
            raise ValueError("Treegramslist need to be given!")

        # # check if all the labelled mergegrams for the treegrams
        # # have been computed        
        # if np.any([tg is None for tg in self.treegrams.treegrams]):
        #     self.treegrams.compute_labelled_mergegram()
        if self.algorithm == 'Alg1':
            self.mergegram = \
                facegram_alg1(self.treegramslist,
                    num_taxa=np.shape(self.treedistances)[-1],
                    container=alg1container)
        elif self.algorithm == 'Alg2':
            self.mergegram = facegram_alg2(self.treegramslist,
                                           self.treedistances)
        else:
            raise ValueError("Algorithm not implemented")

        # self.mergegram = format_mergegram(self.mergegram, self.delete_infinite)
        self.mergegram = dict(sorted(self.mergegram.items(),
                key=lambda x: (x[1][0], x[1][1]-x[1][0])))
        return self.mergegram

    # def compute_mergegram(self):
    #     if self.treegrams is not None:
    #         return(self.compute_join())
    #     else:
    #         raise ValueError("Generqal method hasn't been implemented yet")

    def return_mergegram(self, delete_infinite=None):
        if delete_infinite is not None:
            self.delete_infinite = delete_infinite

        if self.mergegram is None:
            raise ValueError("Mergegram needs to be computed first")

        return format_mergegram(self.mergegram, self.delete_infinite)


def facegram_from_simplexTree(simplextree, num_taxa, tqdm_disable=True):
    filtration = {tuple(x): f for x, f in simplextree.get_filtration()}
    filtration = {k: filtration[k] for k in list(filtration.keys())[::-1]}

    # num_taxa = np.shape(X)[0]
    taxa_sortedcontainers = [SortedSet() for _ in range(num_taxa)]
    taxa_names = {}

    for key, birth in tqdm(filtration.items(),
                           disable=tqdm_disable,
                           total=len(filtration)):
        birth_superset = sorted_intersection_min(key, taxa_sortedcontainers)
        # print(key, birth, birth_superset)
        if birth_superset is None:
            pass
        elif birth_superset <= birth:
            continue
        taxa_names[(birth, hash(key))] = key
        for k in key:
            # We should be using numbering of the element for a given filtration
            # value, but we rather use hashes - we only need to have different
            # values in the second position (whatever they are)
            taxa_sortedcontainers[k].add((birth, hash(key)))

    mergegram = []
    for sigma, face in taxa_names.items():
        birth_superset = sorted_intersection_min(face,
                            taxa_sortedcontainers,
                            sigma)
        if birth_superset is not None:
            mergegram.append((sigma[0], birth_superset))
        else:
            mergegram.append((sigma[0], np.inf))

    return mergegram