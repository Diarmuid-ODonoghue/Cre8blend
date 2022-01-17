# DFS is a heuristic algorithm that presumes that edges with greater total significance
# should be given a greater priority than edges with a smaller total significance
# It explores the edge-space of graph-subgraph near isomorphism.
# loosely inspired by Nawaz, Enscore and Ham (NEH) algorithm
# local optimisation, near the global optimum.
# edges described by a 4-tuple of in/out degrees from a di-graph. 2 edges are compared by Wasserstein metric.
# I believe its an admissable heuristic! A narrow search space is explored heuristically.
#
#import networkx as nx
import sys
import math
#import Map2Graphs.mode as mode
#from Map2Graphs import mode
#import numpy as np

from nltk.stem import WordNetLemmatizer
#from nltk.corpus import wordnet as wn
wnl = WordNetLemmatizer()
import pdb

global mode
global term_separator
global max_topology_distance
global semantic_factors

max_topology_distance = 7 # in terms of a node's in/out degree
mode = "English"
#mode = 'code'
if mode == "English":
    term_separator = "_" #Map2Graphs.term_separator
else:
    term_separator = ":"
semantic_factors = True

from sense2vec import Sense2Vec
#s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
query = "drive|VERB"
assert query in s2v
s2v.similarity(['run' + '|VERB'], ['eat' + '|VERB'])
global s2v_cache
s2v_cache = dict()
#query = "can|VERB"
#assert query in s2v
vector = s2v[query]
freq = s2v.get_freq(query)
#most_similar = s2v.most_similar(query, n=3)
#print(most_similar)
def find_nearest(vec):
    for key, vec in s2v.items():
        print(key, vec)


beam_size = 3  # beam breadth for beam search
epsilon = 10
current_best_mapping = []
bestEverMapping = []

def __init__(G1, G2):
    self.G1 = G1
    self.G2 = G2
    self.core_1 = {}
    self.core_2 = {}
    self.mapping = self.core_1.copy()

def sim_to_dist(n):
    return 1 -1/(0.000001 +n)

def MultiDiGraphMatcher(targetGraph, souceGraph):
    generate_and_explore_mapping_space(targetGraph, souceGraph)

def is_isomorphic():
    print(" DFS.is_isomorphic() ")

def generate_and_explore_mapping_space(targetGraph, sourceGraph):
    global current_best_mapping
    global bestEverMapping
    global semantic_factors
    current_best_mapping = []
    bestEverMapping = []
    if targetGraph.number_of_nodes() == 0 or sourceGraph.number_of_nodes() == 0:
        return [], 0
    global beam_size
    global epsilon
    source_preds = return_sorted_predicates(sourceGraph)
    target_preds = return_sorted_predicates(targetGraph)
    candidate_sources = []
    for t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj in target_preds:
        best_distance, composite_distance, best_subj, best_reln, best_obj \
            = sys.maxsize, sys.maxsize, "nil", "nil", "nil"
        beam = 0
        alternate_candidates = alternates_confirmed = []
        for s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj in source_preds:
            beam += 1
            topology_dist = euclidean_distance(t_subj_in, t_subj_out, t_obj_in, t_obj_out,
                                               s_subj_in, s_subj_out, s_obj_in, s_obj_out)
            if topology_dist > max_topology_distance:
                continue
            elif (t_subj == t_obj) and (s_subj != s_obj):  # what if one is a self-map && the other not ...
                continue
            topology_dist = euc_to_unit(topology_dist)
            if semantic_factors:
                reln_dist = max(0.05, relational_distance(t_reln, s_reln))
                subj_dist = conceptual_distance(t_subj, s_subj)
                obj_dist = conceptual_distance(t_obj, s_obj)
            else:
                reln_dist = subj_dist = obj_dist = 1
            h_prime = scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, sourceGraph, targetGraph)/10
            composite_distance = reln_dist * (topology_dist + h_prime) + (subj_dist + obj_dist)*10
            composite_distance = (reln_dist + subj_dist + obj_dist) * (topology_dist + h_prime) + (subj_dist + obj_dist)*20
            if composite_distance < best_distance:
                best_distance = composite_distance
            alternate_candidates.append([s_subj_in, s_subj_out, s_obj_in, s_obj_out, \
                                         s_subj, s_reln, s_obj, composite_distance])
        alternate_candidates.sort(key=lambda x: x[7])
        print("", end="")
        if len(alternate_candidates)>0:
            alternates_confirmed = []
            for x in alternate_candidates:
                if abs( x[7] - best_distance) < epsilon:
                    alternates_confirmed.append(x) # flat list of sublists
        alternates_confirmed = alternates_confirmed[:beam_size]  # consider only best beam_size options
        candidate_sources.append(alternates_confirmed) #...(alternates_confirmed[0])
        print("", end="")
    reslt = explore_mapping_space(target_preds, candidate_sources, [])
    print()
    #zz = evaluateMapping(bestEverMapping[0])
    print("RESULT ", len(bestEverMapping), end="  ")
    return bestEverMapping, len(bestEverMapping)


def return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj, t_preds, s_preds): # {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}})
    result_list = []
    t_in_deg, h_prime, rel_dist = 0, 0, 0
    for in_t_nbr, foovalue in t_preds:  # find MOST similar S & T pair
        in_t_rel = foovalue[0]['label']
        t_in_deg = targetGraph.in_degree[t_subj]
        for in_s_nbr, foovalue2 in s_preds:
            in_s_rel = foovalue2[0]['label']
            rel_dist = max(relational_distance(in_t_rel, in_s_rel), 0.01)
            s_in_deg = max(0.1, sourceGraph.in_degree[s_subj])
            scor = dist_to_sim(rel_dist) * min(max(1, t_in_deg), s_in_deg)
            result_list.append([scor, in_t_rel, in_s_rel, in_t_nbr, in_s_nbr])
    result_list.sort(key=lambda x: x[0], reverse=True)  # sum the first out_degree numbers
    h_prime = sum(i[0] for i in result_list[:t_in_deg])
    return h_prime, dist_to_sim(rel_dist)


def return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj, t_preds, s_preds): # {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}})
    result_list = []
    t_out_deg, h_prime, rel_dist = 0, 0, 0
    for out_t_nbr, foovalue in t_preds:
        in_t_rel = foovalue[0]['label']
        t_out_deg = targetGraph.out_degree[t_obj]
        for out_s_nbr, foovalue2 in s_preds:
            in_s_rel = foovalue2[0]['label']
            rel_dist = relational_distance(in_t_rel, in_s_rel)
            s_out_deg = max(0.1, sourceGraph.out_degree[s_obj])
            scor = dist_to_sim(rel_dist) * min(max(1, targetGraph.out_degree[t_obj]), s_out_deg)
            result_list.append([scor, in_t_rel, in_s_rel, out_t_nbr, out_s_nbr])
    result_list.sort(key=lambda x: x[0], reverse=True)  # sum the first out_degree numbers
    h_prime = sum(i[0] for i in result_list[:t_out_deg])
    return h_prime, dist_to_sim(rel_dist)


def scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, sourceGraph, targetGraph, reslt=[], level=1):
    if level == 0:
        return 0 #apply sum iterator over reslt
    reln_sim = dist_to_sim( relational_distance(t_reln, s_reln) )
    subj_dist = conceptual_distance(t_subj, s_subj)
    obj_dist = conceptual_distance(t_obj, s_obj)
    best_in_links, in_rel_sim = return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj,
                                         targetGraph.pred[t_subj].items(), sourceGraph.pred[s_subj].items())
    best_out_links, out_rel_sim = return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj,
                                         targetGraph.succ[t_obj].items(), sourceGraph.succ[s_obj].items())
    reln_val = reln_sim + in_rel_sim + out_rel_sim + subj_dist + obj_dist
    reln_val= max(0.01, reln_val)
    return reln_val

def sim_to_dist(sim):
    return 1-sim

def dist_to_sim(dis):
    return (dis-1)*-1

def euc_dist_to_sim(sim):
    return 1/(1+(sim**2))

def euc_to_unit(dist):
    return 1 - (dist-1)*-1


def explore_mapping_space_DEPRECATED(t_preds_list, s_preds_list, globl_mapped_predicates):
    """ Map the next target pred, by finding a mapping from the sources"""
    global bestEverMapping
    if len(globl_mapped_predicates) > len(bestEverMapping): # compare scores, not lengths?
        bestEverMapping = globl_mapped_predicates
    if t_preds_list == [] or s_preds_list == []:
        #bestEverMapping.append(globl_mapped_predicates)
        return globl_mapped_predicates
        #recursive_search(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
    elif t_preds_list != [] and s_preds_list[0] == []:
        explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates) # skip t_pred
        explore_mapping_space(t_preds_list, s_preds_list[1:], globl_mapped_predicates)
        #bestEverMapping = globl_mapped_predicates  # works for Greedy solution
        #return globl_mapped_predicates
    elif t_preds_list == [] or s_preds_list == []:
        bestEverMapping.append(globl_mapped_predicates)
        return globl_mapped_predicates
        #recursive_search(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
    t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj = t_preds_list[0]
    if type(s_preds_list[0]) is int: # Error
        s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj = s_preds_list[0]
        sys.exit(22)
    if type(s_preds_list[0]) is list:
        if type(s_preds_list[0][0]) is list:  # alternates list
            currentOptions = s_preds_list[0]
        else:
            currentOptions = [s_preds_list[0]]  # wrap the single pred within a list
    else:
        sys.exit("DFS.py Line-221 Error  s_preds_list malformed :-(")
    candidates = []
    for singlePred in currentOptions: # score options
        s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj, score = singlePred
        mapped_subjects = check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates)
        mapped_objects = check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates)
        unmapped_subjects = check_if_unmapped(t_subj, s_subj, globl_mapped_predicates)
        unmapped_objects = check_if_unmapped(t_obj, s_obj, globl_mapped_predicates)
        # wass = quick_Wasserstein([t_subj_in, t_subj_out, t_obj_in, t_obj_out], # 4 tuple
        #                         [s_subj_in, s_subj_out, s_obj_in, s_obj_out])
        # if wass > max_topology_distance:  # too topologically different
        #    continue
        # scor = max(0.01, wass)
        # relational_difference_score = targetGraph.number_of_edges(t_subj, t_obj) - \
        #                              sourceGraph.number_of_edges(s_subj, s_obj)
        if mapped_subjects:     # Bonus for intersecting with the current mapping
            score = score / 2
        elif not unmapped_subjects:
            scor = max_topology_distance
            continue
        if mapped_objects:
            score = score / 2
        elif not unmapped_objects:
            score = max_topology_distance
            continue
        # rel_dis = relational_distance(t_reln, s_reln)
        # scor = scor + rel_dis
        if s_subj == s_obj or t_subj == t_obj: # quick test for one reflexive relation
            if (s_subj == s_obj and t_subj != t_obj) or (s_subj != s_obj and t_subj == t_obj):
                score = max_topology_distance
            else:
                candidates = candidates + [[score, s_subj, s_reln, s_obj]]
        else:
            candidates = candidates + [ [score, s_subj, s_reln, s_obj] ]
    candidates.sort(key=lambda x: x[0])
    zz=0
    for dist, s_subj, s_reln, s_obj in candidates: # assign best
        rel_dist = relational_distance(t_reln, s_reln)
        #if relational_distance(t_reln, s_reln) > 0.9: #allow novel relations
        #    continue
        candidatePair = [[t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj], rel_dist]
        if (check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) or \
                 check_if_unmapped(t_subj, s_subj, globl_mapped_predicates) ):
            if check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates) or \
                check_if_unmapped(t_obj, s_obj, globl_mapped_predicates):
                    return explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates + [candidatePair])
        else: # candidate_pair incompatible with current mapping
            return explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
            # then consider next branch in search tree


def explore_mapping_space(t_preds_list, s_preds_list, globl_mapped_predicates):
    """ Map the next target pred, by finding a mapping from the sources"""
    global bestEverMapping
    if len(globl_mapped_predicates) > len(bestEverMapping):  # compare scores, not lengths?
        bestEverMapping = globl_mapped_predicates
    if t_preds_list == [] or s_preds_list == []:
        return globl_mapped_predicates
    elif t_preds_list != [] and s_preds_list[0] == []:
        explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates) # skip t_pred
        explore_mapping_space(t_preds_list, s_preds_list[1:], globl_mapped_predicates)
        #bestEverMapping = globl_mapped_predicates  # works for Greedy solution
        #return globl_mapped_predicates
    elif t_preds_list == [] or s_preds_list == []:
        bestEverMapping.append(globl_mapped_predicates)
        return globl_mapped_predicates
        #recursive_search(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
    t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj = t_preds_list[0]
    if type(s_preds_list[0]) is int: # Error
        sys.exit(22)
    if type(s_preds_list[0]) is list:
        if s_preds_list[0] == []:
            if len(s_preds_list) > 0:
                if s_preds_list[0] == []:
                    currentOptions = []
                else:
                    currentOptions = s_preds_list[1:][0]
            else:
                currentOptions = []
        elif type(s_preds_list[0][0]) is list:  # alternates list
            currentOptions = s_preds_list[0]
        else:
            currentOptions = [s_preds_list[0]]  # wrap the single pred within a list
    else:
        sys.exit("DFS.py Line-221 Error  s_preds_list malformed :-(")
    candidates = []
    for singlePred in currentOptions: # score options
        s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj, score = singlePred
        mapped_subjects = check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates)
        mapped_objects = check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates)
        unmapped_subjects = check_if_unmapped(t_subj, s_subj, globl_mapped_predicates)
        unmapped_objects = check_if_unmapped(t_obj, s_obj, globl_mapped_predicates)
        if mapped_subjects:     # Bonus for intersecting with the current mapping
            score = score / 2
        elif not unmapped_subjects:
            score = max_topology_distance
            continue
        if mapped_objects:
            score = score / 2
        elif not unmapped_objects:
            score = max_topology_distance
            continue
        if s_subj == s_obj or t_subj == t_obj: # quick test for one reflexive relation
            if (s_subj == s_obj and t_subj != t_obj) or (s_subj != s_obj and t_subj == t_obj):
                score = max_topology_distance
            else:
                candidates = candidates + [[score, s_subj, s_reln, s_obj]]
        else:
            candidates = candidates + [ [score, s_subj, s_reln, s_obj] ]
    candidates.sort(key=lambda x: x[0])
    for dist, s_subj, s_reln, s_obj in candidates: # assign best
        candidatePair = [[t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj], dist]
        if (check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) or \
                 check_if_unmapped(t_subj, s_subj, globl_mapped_predicates) ):
            if check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates) or \
                check_if_unmapped(t_obj, s_obj, globl_mapped_predicates):
                    return explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates + [candidatePair])
        else:  # candidate_pair incompatible with current mapping
            return explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
            # then consider next branch in search tree


def checkIfUnmapped_DEPRECATED(t_subj, s_subj, globl_mapped_predicates):
    "accepts 2 full predicates [t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj]  ...]"
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:  #second_head()
        t_s, t_v, t_o = x[0]
        s_s, s_v, s_o = x[1]
        if t_subj == t_s or t_subj == t_o:
            return False
        elif s_subj == s_s or s_subj == s_o:
            return False
    return True
#assert( checkIfUnmapped(1, 11, [ [1,2,3], [11,12,13] ])  ) == False
#checkIfUnmapped(1, 11, [[1,11]])


def check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates):
    if globl_mapped_predicates == []:
        return False
    for x in globl_mapped_predicates:  #second_head()
        t_s, t_v, t_o = x[0]
        s_s, s_v, s_o = x[1]
        if t_subj == t_s and s_subj == s_s:
            return True
        elif t_subj == t_o and s_subj == s_o:
            return True  # and if its a commutitative relation...
        elif t_subj == s_s and s_subj == t_o:
            return True
        elif t_subj == s_o and s_subj == t_s:
            return True
    return False


def check_if_unmapped(t_subj, s_subj, globl_mapped_predicates):
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:
        t_s, t_v, t_o = x[0]  # target
        s_s, s_v, s_o = x[1]
        if t_subj == t_s or s_subj == s_s:
            return False
        elif t_subj == t_o or s_subj == s_o:
            return False
        else:
            return True
    return True

def check_if_unmapped_DEPRECATED(t_subj, s_subj, globl_mapped_predicates):
    if t_subj == 'shot':
        x=0
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:
        if t_subj in x[0] or t_subj in x[1]:
            return False
        elif s_subj in x[0] or s_subj in x[1]:
            return False
    return True


def evaluateMapping(globl_mapped_predicates):
    "[[['hawk_Karla_she', 'saw', 'hunter'], ['hawk_Karla_she', 'know', 'hunter'], 0.715677797794342], [['h"
    mapping = dict()
    relatio_structural_dist = 0
    for t_pred,s_pred,val in globl_mapped_predicates:  # t_pred,s are full predicates
        relatio_structural_dist += val
        if t_pred==[] or s_pred==[]:
            continue
        # print("{: >20} {: >10} {: >20}".format(s, v, o, "    ==    "))
        if t_pred[0] not in mapping.keys() and s_pred[0] not in mapping.values():
            mapping[t_pred[0]] = s_pred[0]
            mapping[t_pred[2]] = s_pred[2]
        elif mapping[t_pred[0]] != s_pred[0]:
            print("\n-- Mis-Mapping 1 ", s_pred, t_pred, "         ")
            #sys.exit(" Mis-Mapping 1 in DFS ")
        elif t_pred[2] not in mapping.keys() and s_pred[2] not in mapping.values():
            print("-- Mis-Mapping 2 ", t_pred[2], s_pred[2], "       ", mapping) #, end=" ")
            #print(mapping[t_pred[2]], end="   ")
            mapping[t_pred[2]] = s_pred[2]
        elif  t_pred[2] in mapping.keys() and mapping[t_pred[2]] != s_pred[2]: #and
            print("\n*** Mis-Mapping 2", s_pred[2], t_pred[2], "*** Possible Reflexive relation ***")
            #sys.exit("Mis-Mapping 2.2 in DFS ")
        else:
            print("-- Mis-Mappping 3 ", t_pred[2], s_pred[2], "       ", mapping)
    print("     MAPPING", len(mapping), relatio_structural_dist, "  ", mapping, end="     \t")
    return relatio_structural_dist


def relational_distance(t_reln, s_reln):   # using s2v sense2vec
    global s2v_cache
    s_reln = s_reln.split(term_separator)[0]
    t_reln = t_reln.split(term_separator)[0]
    if s_reln == t_reln:
        return 0.01
    elif second_head(t_reln) == second_head(s_reln):
        return 0.1
    elif t_reln+"-"+s_reln in s2v_cache:
        return s2v_cache[t_reln+"-"+s_reln]
    elif s_reln+"-"+t_reln in s2v_cache:
        return s2v_cache[s_reln+"-"+t_reln]
    else:
        if s2v.get_freq(t_reln + '|VERB') is None or s2v.get_freq(s_reln + '|VERB') is None:
            t_root = wnl.lemmatize(t_reln)
            s_root = wnl.lemmatize(s_reln)
            if s2v.get_freq(t_root + '|VERB') is None or s2v.get_freq(s_root + '|VERB') is None:
                return 1
            else:
                return s2v.similarity([t_root + '|VERB'], [s_root + '|VERB'])
        else:
            sims_core = s2v.similarity([t_reln + '|VERB'], [s_reln + '|VERB'])
            if sims_core >= 0.01:
                sims_core = 1 - sims_core
            else:
                sims_core = 1.0
            s2v_cache[t_reln+"-"+s_reln] = sims_core
            return sims_core


def second_head(node):
    global term_separator
    if not (isinstance(node, str)):
        return ""
    else:
        lis = node.split(term_separator)
        if len(lis) >= 2:
            wrd = lis[1].strip()
        else:
            wrd = lis[0].strip()
        return wrd


def linear_distance_string(strg): # based on graph properties in/out degrees
    t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj, \
        s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj = strg
    z = math.sqrt((t_subj_in - s_subj_in) ** 2 + (t_subj_out - s_subj_out) ** 2 +
              (t_obj_in - s_obj_in) ** 2 + (t_obj_out - s_obj_out) ** 2)
    return z + 0.1


def euclidean_distance(t_subj_in, t_subj_out, t_obj_in, t_obj_out,
                       s_subj_in, s_subj_out, s_obj_in, s_obj_out):
    z = math.sqrt( (t_subj_in - s_subj_in)**2  + (t_subj_out - s_subj_out) ** 2 +
                   (t_obj_in - s_obj_in) **2   + (t_obj_out - s_obj_out) **2 )
    return z + 0.001


def conceptual_distance(str1, str2):
    "for simple conceptual similarity"
    global term_separator
    if str1 == str2:
        return 0.01
    str1 = str1.split(term_separator)
    str2 = str2.split(term_separator)
    if str1[0] == str2[0]: # intersection over difference
        inter_sectn = list( set(str1) & set(str2))
        if len(inter_sectn) > 0:
            return min(0.1, 0.2 ** len(inter_sectn))
        else:
            return 0.25
    elif mode == "English":
        #print(str1, str2, end=" ")
        if s2v.get_freq(str1[0] + '|NOUN') is None or \
            s2v.get_freq(str2[0] + '|NOUN') is None:
            return 10
        else:
            sim_score = s2v.similarity([str1[0] + '|NOUN'], [str2[0] + '|NOUN'])
            #print(sim_score, end="  ")
            return sim_score
    else:
        return 2.0

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def return_sorted_predicates(grf):
    edge_list = grf.edges.data("label")
    zz = [(n, d) for n, d in grf.degree()]
    yy = sorted(zz, key=lambda x: x[1], reverse=True)
    pred_list = []
    for (s, o, v) in edge_list:
        pred_list.append( [grf.in_degree(s), grf.out_degree(s), grf.in_degree(o), grf.out_degree(o),s, v, o] )
    z = sorted(pred_list, key=lambda x: x[1], reverse=True)
    return sorted(z, key=lambda x: x[0]+x[1]+x[2]+x[3], reverse=True)


def findArity(nod, tupl_list):
        for n, x in tupl_list:
            if n == nod:
                return x

#pred_comparison_vector = subj_in, subj_out, reln, obj_in, obj_out   (reln1 to reln2)

def quick_Wasserstein(a, b):  # inspired by Wasserstein distance (quick Wasserstein approximation)
    a_prime = sorted(list(  set(a) - set(b) ))
    b_prime = sorted(list(  set(b) - set(a) ))
    if len(a_prime) < len(b_prime): #longer list is a_prime
        temp = b_prime.copy()
        b_prime = a_prime.copy()
        a_prime = temp.copy()
    sum1 = sum(abs(i-j) for i, j in zip(a_prime, b_prime))
    b_len = len(b_prime)
    sum2 = sum(a_prime[b_len:])
    return sum1 + sum2


# #########

def return_best_candidate(w1, target, PoS_tag):
    candidates1 = s2v.most_similar(w1+"|"+PoS_tag, n=15)
    # 3000 most common english words
    candidates1_list = []
    for (x,y) in candidates1:
        candidates1_list.append([ s2v.similarity([x], [target+"|"+PoS_tag]), x ])
    candidates1_list
    res = sorted(candidates1_list, key=lambda x: x[0])[::-1]
    return res


def midpoint_between_words(w1, w2, PoS_tag):
    query1 = w1
    query2 = w2
    assert query1+"|"+PoS_tag in s2v
    assert query2+"|"+PoS_tag in s2v
    count = 0
    score1 = []
    score2 = []
    previous_best = sys.maxsize
    while count < 2:
        simlr1_to_target = return_best_candidate(query1, w2, PoS_tag)
        simlr1_to_home = return_best_candidate(query1, w1, PoS_tag)
        simlr2_to_target = return_best_candidate(query2, w1, PoS_tag)
        simlr2_to_home = return_best_candidate(query2, w2, PoS_tag)
        for (a, b) in zip(simlr1_to_target, simlr1_to_home):
            if b[1].split("|")[0] != w1 and b[1].split("|")[0] != w2:
                score1.append( [b[0] + a[0] + abs(a[0]-b[0]), a[1] ])
                # score1.append([b[0] + a[0], a[1]] )
                # score1.append( [ (abs(b[0] - a[0])), a[1]] )
        for (a, b) in zip(simlr2_to_target, simlr2_to_home):
            if b[1].split("|")[0] != w2 and b[1].split("|")[0] != w1:
                # score2.append([b[0] + a[0], a[1]] )
                score2.append( [ (abs(b[0] - a[0])), a[1] ] )
        midpoint1_estimate = sorted(score1, key=lambda x: abs(x[0]))  # [::-1]
        midpoint2_estimate = sorted(score2, key=lambda x: abs(x[0]))  # [::-1]
        big_list = sorted(midpoint1_estimate + midpoint2_estimate, key=lambda x: abs(x[0]))
        if big_list[0][0] < previous_best:
            previous_best = big_list[0][0]
            best_answer_so_far = big_list[0][1]
        query1 = midpoint1_estimate[0][1].split("|")[0]
        query2 = midpoint2_estimate[0][1].split("|")[0]
        count += 1
        print("LOOP: ", best_answer_so_far, previous_best)
    rslt1 = sorted(big_list, key=lambda x: abs(x[0]))
    print(":::", rslt1[0:5])
    return rslt1


def generalise_word(wrd):
    from nltk.corpus import names, stopwords, words
    words.words('en-basic')  # 850 words

# print(" s2v.similarity(man+|+NOUN, drive+|+VERB)  ", s2v.similarity("man"+"|"+"NOUN", "drive"+"|"+"VERB") )

# print(midpoint_between_words("dog", "cat", "NOUN"))
# print(midpoint_between_words("drive", "fly", "VERB"))

# print( s2v.similarity("man"+"|"+"NOUN", "drive"+"|"+"VERB") )