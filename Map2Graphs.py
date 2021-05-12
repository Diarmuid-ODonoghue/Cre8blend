import networkx as nx
import isomorphvf2CB  # VF2 My personal Modified VF2 variant
import csv
import pprint
import numpy  # as np
import matplotlib.pyplot as plt
import errno
import webbrowser
import json
import os
import os.path
from os import path
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from nltk.corpus import words as nltkwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
import time
import math
from tkinter import *


#from loguru import logger
#import snoop
#import heartrate

# import DFS # in if statement below
from itertools import count, product
from multiprocessing import Process, Queue, Manager
# import sys
# from networkx.algorithms import isomorphism as isomorphvf2CB
# import DFS as isomorphvf2CB #my personal DFS based graph matching
# import ConceptNetElaboration as CN
# import pylab
# import operator
# import subprocess
# from cacheout import Cache
# import requests

from sense2vec import Sense2Vec

s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
#s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
# query = "drive|VERB"
# assert query in s2v

# ######################################################
# ############## Global Variables ######################
# ######################################################

# nx.OrderedMultiDiGraph()  #loosely-ordered, mostly-directed, somewhat-multiGraph, self-loops, parallel edges
global localBranch
global invalid_graphs
global algorithm
global coalescing_completed
global max_graph_size
global list_of_mapped_preds
global mapping_run_time

run_Limit = 50000  # 5 # Stop after this many texts
skip_prepositions = True  # False
# skip_prepositions = False

algorithm = "DFS"
algorithm = "ismags"
#algorithm = "vf2"
if algorithm == "DFS":
    import DFS

mode = 'English'
mode = 'code'
if mode == 'code':
    filetypeFilter = '.csv'  # 'txt.S.csv'
    term_separator = ":"  # "_"  hawk_he OR Block:Else: If
    localBranch = "/c-sharp-id-num/"
    # localBranch = "/AllCSVs/"
    # localBranch = "/C-Sharp Data/"
    # localBranch = "/Java Data/"
    localBranch = "/TS1-TS4/"
else:
    filetypeFilter = 'dcorf.csv'  # 'txt.S.csv'
    filetypeFilter = '.csv'  # 'txt.S.csv'
    mode == 'English'
    term_separator = "_"  # "_"  hawk_he
    localBranch = "/test/"  # "test/" #Text2ROS.localBranch #""
    #localBranch = "/iProva/"
    # localBranch = "/Covid-19/"
    localBranch = "/Psychology Data/"
    localBranch = "/MisTranslation Data/"
    # localBranch = "/Killians Summaries/"
    # localBranch = "/SIGGRAPH ROS/"
    #localBranch = "/Sheffield-Plagiarism-Corpus/"

basePath = "C:/Users/user/Documents/Python-Me/data"
#basePath = "C:/Users/dodonoghue/Documents/Python-Me/data"
# basePath = dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')
localPath = basePath + localBranch
CSVPath = localPath + "CSV Output/"  # Where you want the CSV file to be produced
CachePath = basePath + localBranch + "/Cache.txt"  # Where you saved the Cache txt file
CSVsummaryFileName = CSVPath + "summary.csv"
analogyFileName = CSVPath + "something.csv"
list_of_mapped_preds = []


sourceFiles = os.listdir(localPath)
all_csv_files = [i for i in sourceFiles if i.endswith(filetypeFilter)]
# all_csv_files = [i for i in sourceFiles if i.endswith('txt.S.csv') if ("code" in i) and (i.endswith('.csv'))]
# all_csv_files = [i for i in sourceFiles if i.endswith(filetypeFilter) and i.startswith('S')]
all_csv_files.sort()#reverse=True)
print(len(all_csv_files), "  ", all_csv_files)
print("\nINPUT:", localPath)
print("OUTPUT:", CSVPath)


max_graph_size = 150  # see prune_peripheral_nodes(graph)
invalid_graphs = 0
targetGraph = nx.MultiDiGraph()  # <- targetFile
sourceGraph = nx.MultiDiGraph()  # <- sourceFile
temp_graph2 = nx.OrderedMultiDiGraph()  # create ordered graphs. Subsequently treat as unordered.
numberOfTimeOuts = 0
list_of_inferences = []
LCSLlist = []
list_of_mapped_preds = []
listOfSources = []
relationMapping = []
WN_cache = {}
CN_dict = {}
mapping_run_time = 0

# semcor_ic = wordnet_ic.ic('ic-semcor.dat')
brown_ic = wordnet_ic.ic('ic-brown.dat')
vbse = 0  # VerBoSE mode for error reporting and tracing
wnl = WordNetLemmatizer()
wn_lemmas = set(wordnet.all_lemma_names())
pp = pprint.PrettyPrinter(indent=4)

global arrow
arrow = " <-> "

########################################################
############## File Infrastructure #####################
########################################################


htmlBranch = basePath + localBranch + "FDG/"

CN_file = basePath + "/ConceptNetdata.csv"


print("CSV input files: ", len(all_csv_files))
print("\nMode=", mode, "  Term Separator=", term_separator)

commutativeVerbList = ['and', 'beside', 'near']  # x and y  ==>  y and x

# pronouns and pronomial adjectives
pronoun_list = ["all", "another", "any", "anybody", "anyone", "anything", "as", "aught", "both", "each",
                "each other", "either", "enough", "everybody", "everyone", "everything", "few", "he", "her",
                "hers", "herself", "him", "himself", "his", "I", "idem", "it", "its", "itself", "many", "me ",
                " mine", "most", "my", "myself", "naught", "neither", "no one", "nobody", "none", "nothing",
                "nought", "one", "one another", "other", "others", "ought", "our", "ours", "ourself",
                "ourselves", "several", "she", "some", "somebody", "someone", "something", "somewhat",
                "such", "suchlike", "that", "thee", "their", "theirs", "theirself", "theirselves", "them",
                "themself", "themselves", "there", "these", "they", "thine", "this", "those", "thou", "thy",
                "thyself", "us", "we ", " what", "whatever", "whatnot", "whatsoever", "whence", "where",
                "whereby", "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wherever", "wheresoever",
                "whereto", "whereunto", "wherewith", "wherewithal", "whether", "which", "whichever",
                "whichsoever", "who", "whoever", "whom", "whomever", "whomso", "whomsoever", "whose",
                "whosever", "whosesoever", "whoso", "whosoever", "ye", "yon", "yonder", "you", "your", "yours",
                "yourself", "yourselves"]


def prepositionTest(word):
    prepList = ['of', 'for', 'at', 'on', 'as', 'by', 'in', 'to', 'from',
                'into', 'through', 'toward', 'with']
    return word in prepList


# #################################################################################################################
# ##################################### Process Input #############################################################
# #################################################################################################################

def extend_as_set(l1, l2):
    result = []
    if len(l1) >= len(l2):
        result.extend(x for x in l1 if x not in result)
        donor = l2
    else:
        result.extend(x for x in l2 if x not in result)
        donor = l1
    result.extend(x for x in donor if x not in result)
    coref_terms = '_'.join(word for word in result)
    return reorganise_coref_chain(coref_terms)


# extend_as_set(['hawk', 'she'], ['hawk', 'it'])
# extend_as_set(['hunter', 'He'], ['hunter', 'he', 'him'])


def merge_concept_chains(c1, c2):  # simple set merge
    global term_separator
    c1 = c1.strip().split(term_separator)
    c2 = c2.strip().split(term_separator)
    if len(c1) >= len(c2):
        res = set(c1 + c2)
    else:
        res = set(c2 + c1)
    return reorganise_coref_chain(res)


def graph_contains_proper_noun(propNoun1):
    global temp_graph2
    flag = False
    zzz = propNoun1
    pred_list = temp_graph2.nodes()
    for subj in pred_list:
        for wrd in subj.split(term_separator):
            if propNoun1 == wrd:
                zzz = subj
    return zzz


def pre_existing_node(graph_name, inNoun):
    """ Merges inNoun into existing nodes from that graph - if applicable """
    if inNoun == "NOUN":  # skip any header information
        return inNoun
    global term_separator
    global temp_graph2
    global wn_lemmas
    flag = False
    inNoun_head = head_word(inNoun)
    extendedNoun = inNoun
    in_propN = contains_proper_noun(inNoun)
    # dud = temp_graph2.nodes()
    for graph_node in temp_graph2.nodes():
        if inNoun == graph_node:
            flag = False
            break
        elif in_propN != False:  # inNoun contains a PROPER noun
            if (in_propN == contains_proper_noun(graph_node)) and not (inNoun is graph_node):  #
                flag = True  # ProperNoun parts must be identical
                break
            elif inNoun_head == head_word(graph_node) and not (
                    inNoun == graph_node):  # and not(is_pronoun(inNoun_head)):
                flag = True  # same head, not pronoun!
                break
            elif inNoun_head.lower() == head_word(graph_node).lower() and not (
                    inNoun == graph_node):  # and not(is_pronoun(inNoun_head)):
                flag = True  # same head, not pronoun!
                break
        elif inNoun_head == head_word(graph_node) and (inNoun != graph_node) \
                and inNoun_head != False:  # identical head noun
            flag = True
            break
        elif inNoun_head.lower() == head_word(graph_node).lower() and (inNoun != graph_node) \
                and inNoun_head != False:  # identical head noun
            flag = True
            break
        else:
            # tmp = head_word(graph_node)
            if inNoun_head != False and head_word(graph_node) != False:  # identicality previously tested
                a = wnl.lemmatize(inNoun_head)
                b = wnl.lemmatize(head_word(graph_node))
                if a == b:
                    flag = True
                    break
    if flag:
        if inNoun != graph_node:
            extendedNoun = extend_as_set(inNoun.split(term_separator),
                                         graph_node.split(term_separator))  # !!!!!!!!!!!!!!!!!!!!
            extendedNoun = reorganise_coref_chain(extendedNoun)
            if graph_node != extendedNoun:
                # print(inNoun,"+", graph_node, ">>", extendedNoun, end="     ")
                remapping = {graph_node: extendedNoun}
                graph_name = nx.relabel_nodes(graph_name, remapping, copy=False)
                graph_name.nodes[extendedNoun]['label'] = extendedNoun
            else:
                print(inNoun, ">", extendedNoun, end="     ")
    return extendedNoun


def build_graph_from_csv(file_name):
    """ Includes eager concept fusion rules. Enforces  noun_properNoun_pronoun"""
    global temp_graph2  # an ORDERED multi Di Graph, converted at the end
    global term_separator
    global invalid_graphs
    fullPath = localPath + file_name
    unknownCounter = 1
    # print("BGC")
    with open(fullPath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp_graph2.clear()
        temp_graph2.graph['Graphid'] = file_name
        try:
            previous_subj = last_v = previous_obj = ""
            for row in csvreader:
                # print(row)
                if len(row) == 3:  # subject, verb, obj
                    noun1, verb, noun2 = row
                    noun1 = noun1.strip()
                    verb = verb.strip()
                    noun2 = noun2.strip()
                    if skip_prepositions and prepositionTest(verb):
                        continue
                    if noun1.lower() == 'unknown':
                        noun1 = chr(ord('@') + unknownCounter) + "_" + "unknown"  # Unique proper nouns
                        unknownCounter += 1
                    if noun2.lower() == 'unknown':
                        noun2 = chr(ord('@') + unknownCounter) + "_" + "unknown"
                        unknownCounter += 1
                    if (noun1 == previous_subj) and (noun2 == previous_obj) and prepositionTest(verb):
                        verb = last_v + "_" + verb
                        print(verb, end=" ")
                        temp_graph2.remove_edge(noun1, noun2)
                    elif noun1 == "NOUN":  # skip header information
                        continue  # Coreference chains
                    if len(noun1.split(term_separator)) > 1:
                        noun1 = parse_new_coref_chain(noun1)
                    if len(noun2.split(term_separator)) > 1:
                        noun2 = parse_new_coref_chain(noun2)
                    noun1 = pre_existing_node(temp_graph2, noun1)
                    if not (noun1 in temp_graph2.nodes()):
                        temp_graph2.add_node(noun1, label=noun1)
                    noun2 = pre_existing_node(temp_graph2, noun2)
                    if not (noun2 in temp_graph2.nodes()):
                        temp_graph2.add_node(noun2, label=noun2)
                    if head_word(noun1) == head_word(noun2):  # noun, noun2  INTERACTIONS Remove prounouns first?
                        noun1 = pre_existing_node(temp_graph2, noun1)
                    elif len(set(noun1.split(term_separator)).intersection(set(noun2.split(term_separator)))) > 0:
                        noun1 = pre_existing_node(temp_graph2, noun1)
                    temp_graph2.add_edge(noun1, noun2, label=verb)
                elif len(row) == 4:  # methodName, subject, verb, obj
                    methodName, noun1, verb, noun2 = row
                    noun1 = noun1.strip()
                    verb = verb.strip()
                    noun2 = noun2.strip()
                    if noun1.lower() == 'unknown':
                        noun1 = chr(ord('@') + unknownCounter) + "_" + "unknown"  # Unique proper nouns
                        unknownCounter += 1
                    elif noun2.lower() == 'unknown':
                        noun2 = chr(ord('@') + unknownCounter) + "_" + "unknown"
                        unknownCounter += 1
                    if (noun1 == previous_subj) and (noun2 == previous_obj) and prepositionTest(verb):
                        # and is_phrasal_verb(last_v + " " + verb)
                        phrasal_verb = last_v + "_" + verb
                        temp_graph2.remove_edge(noun1, noun2)
                        temp_graph2.add_edge(noun1, noun2, label=phrasal_verb)
                        continue
                    elif noun1 == "NOUN":  # skip any header information
                        continue  # rest of the file contains prepositions
                    elif skip_prepositions and prepositionTest(verb):
                        continue
                    #  Coreference chains
                    if len(noun1.split(term_separator)) > 1:
                        noun1 = parse_new_coref_chain(noun1)
                    if len(noun2.split(term_separator)) > 1:
                        noun2 = parse_new_coref_chain(noun2)

                    noun1 = pre_existing_node(temp_graph2, noun1)
                    if not (noun1 in temp_graph2.nodes()):
                        temp_graph2.add_node(noun1, label=noun1)

                    noun2 = pre_existing_node(temp_graph2, noun2)
                    # what if NEW node2 should replace current version of node1 23-July
                    if head_word(noun1) == head_word(noun2):
                        noun1 = pre_existing_node(temp_graph2, noun1)

                    # VERB - Edge - Predicate
                    if not (noun2 in temp_graph2.nodes()):
                        temp_graph2.add_node(noun2, label=noun2)
                    temp_graph2.add_edge(noun1, noun2, label=verb)
                elif mode == 'code' or len(row) == 6 or row[0:3] == "Type":  # Code Graphs
                    if len(row) == 0:
                        continue
                    elif row[0] == "CodeContracts":  # skip the contracts?
                        break
                    if len(row) == 6:
                        methodName, noun1, verb, noun2, nr1, nr2 = row
                        noun1 = noun1 + ":" + nr1
                        noun2 = noun2 + ":" + nr2
                    elif len(row) == 3:
                        noun1, verb, noun2 = row
                    elif len(row) != 3:
                        print("SQL problem: ", file_name, end="  ")
                        invalid_graphs += 1
                        continue
                    else:
                        noun1, verb, noun2 = row
                    noun1 = noun1.strip()  # remove spaces
                    verb = verb.strip()
                    noun2 = noun2.strip()
                    temp_graph2.add_node(noun1, label=noun1)
                    temp_graph2.add_node(noun2, label=noun2)
                    temp_graph2.add_edge(noun1, noun2, label=verb)
                if mode == 'English':
                    previous_subj = noun1
                    last_v = verb  # phrasal verbs "run_by"
                    previous_obj = noun2
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (CSVPath, csvreader.line_num, e))
    if mode == 'English':
        # print("Coalescing:\n", temp_graph2.nodes(), end="  ")
        # global coalescing_completed
        # coalescing_completed = False
        # iteration_limit = min(temp_graph2.number_of_nodes(), 1) # Remove redundancy
        # while not coalescing_completed and iteration_limit>0:
        final_pass_coalescing()
        #      iteration_limit -=1
        # print("FPC-loop", iteration_limit, end="    ")
    returnGraph = nx.MultiDiGraph(temp_graph2)
    return returnGraph  # results in the canonical  version of the graph :-)


# end of build_graph_from_csv(targetFile)


def parse_new_coref_chain(in_chain):
    """ For silly long coref chains. """
    if mode == 'Code':
        return in_chain
    cnt = in_chain.find(term_separator)
    if cnt <= 0:
        return in_chain
    if cnt <= 4:  # parser works poorly on short noun sequences
        return reorganise_coref_chain(in_chain)
    elif mode == "English":  # parse_coref_subsentence()
        noun_lis = []
        propN_lis = []
        pron_lis = []
        chn = in_chain.split(term_separator)  # "its_warlike_neighbor_Gagrach".split("_")
        strg2 = " ".join(chn)
        chn2 = nltk.pos_tag(word_tokenize(strg2))
        for (tokn, po) in chn2:
            if tokn in ['a', 'the', 'its']:  # remove problematic words from coref phrases
                continue
            elif is_pronoun(tokn) or po == "PRP":
                pron_lis.append(tokn)
            elif po == "NN" or is_noun(tokn):
                noun_lis.append(tokn)
            elif po == "NNP" or is_proper_noun(tokn):
                propN_lis.append(tokn)
        res = noun_lis + propN_lis + pron_lis
        slt = "_".join(res)
        return slt
    return "ERROR - parse_new_coref_chain() "


def final_pass_coalescing():
    global term_separator
    global temp_graph2
    global coalescing_completed
    coalescing_completed = True
    node_list = list(temp_graph2.nodes())
    zz = node_list
    flag = False
    print("\nFPC", end="       ")
    limit = temp_graph2.number_of_nodes()
    for graph_node in zz:  # node_list:  # Final-pass Coalescing
        gn_pn = contains_proper_noun(graph_node)
        for g_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes #in limit
            gn_pn2 = contains_proper_noun(g_node2)
            if not (gn_pn == False) and (gn_pn == gn_pn2) and not (graph_node is g_node2):
                flag = True
                break
            elif head_word(graph_node) == head_word(g_node2) and not (graph_node is g_node2):
                flag = True
                break
        if flag and not (graph_node == g_node2):
            extendedNoun = extend_as_set(graph_node.split(term_separator), g_node2.split(term_separator))
            extendedNoun = reorganise_coref_chain(extendedNoun)
            remapping = {g_node2: extendedNoun}
            print(" R2Map", graph_node, "+", g_node2, "->", extendedNoun, graph_node == g_node2, end="--    ")
            temp_graph2 = nx.relabel_nodes(temp_graph2, remapping, copy=False)
            temp_graph2.nodes[extendedNoun]['label'] = extendedNoun
            if not (graph_node == extendedNoun):
                remapping = {graph_node: extendedNoun}  # merge in graph_node too
                temp_graph2 = nx.relabel_nodes(temp_graph2, remapping, copy=False)
                temp_graph2.nodes[extendedNoun]['label'] = extendedNoun
            flag = False
            coalescing_completed = False
            break
    print("FPC Done.   ", end="")
    return


def list_diff(li1, li2):
    return (list(set(li1) - set(li2)))


# ####################################################################################
# ####################################################################################
# ####################################################################################

def reorganise_coref_chain_DEPRECATED(strg):  # noun-propernoun-pronoun
    global term_separator
    noun_lis = []
    propN_lis = []
    pron_lis = []
    if strg.find(term_separator) < 0:
        slt = strg
    else:
        chan = strg.split(term_separator)
        for tokn in chan:
            if tokn in ['a', 'the', 'its']:  # remove problematic words
                continue
            elif is_pronoun(tokn.lower()):
                pron_lis.append(tokn)
            elif is_noun(tokn):
                noun_lis.append(tokn)
            elif is_proper_noun(tokn):
                propN_lis.append(tokn)
        res = noun_lis + propN_lis + pron_lis
        slt = "_".join(res)
    return slt


def reorganise_coref_chain(strg):  # noun-propernoun-pronoun
    global term_separator
    noun_lis = []
    propN_lis = []
    pron_lis = []
    if strg.find(term_separator) < 0:
        slt = strg
    else:
        chan = strg.split(term_separator)
        for tokn in chan:
            if tokn in ['a', 'the', 'its']:  # remove problematic words
                continue
            elif is_pronoun(tokn.lower()):
                pron_lis.append(tokn)
            elif is_noun(tokn):
                noun_lis.append(tokn)
            elif is_proper_noun(tokn):
                propN_lis.append(tokn)
        res = noun_lis + propN_lis + pron_lis
        slt = "_".join(res)
    return slt


# reorganise_coref_chain("its_warlike_neighbor_Gagrach")
# reorganise_coref_chain("He_hunter_he_him")
# reorganise_coref_chain('hawk_she_Karla')
# reorganise_coref_chain('hawk_Karla_she')


def head_word(term):  # first word before term separator
    global term_separator
    z = term.split(term_separator)[0]
    return z


def is_proper_noun_DEPRECATED(wrd):
    zz = nltk.pos_tag([wrd])
    return ((nltkwords.words().__contains__(wrd) == False)
            and (wn.synsets(wrd) == []))


# is_proper_noun("Karla")   is_proper_noun("karla")

def is_proper_noun(wrd):
    w, tag = nltk.pos_tag([wrd])[0]
    return tag == "NNP"


def contains_proper_noun(wrd):  # wrd may be a coreference chain
    global term_separator
    for wo in wrd.split(term_separator):
        if ((nltkwords.words().__contains__(wo) == False) and (wn.synsets(wo) == [])):
            return wo
    return False


# contains_proper_noun("Karla")   contains_proper_noun("she_karla")

def contains_proper_noun_from_lis(lis):  # wrd may be a coreference chain
    global term_separator
    for wo in lis:
        if ((nltkwords.words().__contains__(wo) == False) and (wn.synsets(wo) == [])):
            return wo
    return False


def is_pronoun(wrd):
    return wrd in pronoun_list


# is_pronoun("he")

def is_noun(wrd):
    zz = wn.synsets(wrd, pos=wn.NOUN)
    return zz != []


# is_noun("a")

# ###############################
# ####### Process Graphs ########
# ###############################


def printEdges(G):  # printEdges(sourceGraph)
    for (u, v, reln) in G.edges.data('label'):
        print('(%s %s %s)' % (u, reln, v))


def returnEdges(G):  # returnEdges(sourceGraph)
    """returns a list of edge names, followed by a printable string """
    res = ""
    for (u, v, reln) in G.edges.data('label'):
        res = res + u + " " + reln + " " + v + '.' + "\n"
        print(reln, end=" ")
    return res


# returnEdges(targetGraph)


def returnEdgesAsList(G):  # returnEdgesAsList(sourceGraph)
    """ returns a list of lists, each composed of triples"""
    res = []
    for (u, v, reln) in G.edges.data('label'):
        res.append([u, reln, v])
    return res


# returnEdgesAsList(targetGraph)

def ps():
    print(returnEdgesAsList(sourceGraph))


def pt():
    print(returnEdgesAsList(targetGraph))


def returnEdgesBetweenTheseObjects(subj, obj, thisGraph):
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, o, relation) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj):
            res.append(relation)
    return res


# returnEdgesBetweenTheseObjects('woman','bus', targetGraph)

def returnEdgesBetweenTheseObjects_predList(subj, obj, pred_list):
    """ returns a list of verbs (directed link labels) between objects - or else [] """
    res = []
    for (s, v, o) in pred_list:
        if (s == subj) and (o == obj):
            res.append(v)
    return res


def predExists(subj, rel, obj, thisGraph):  # predExists('man','drive','car', sourceGraph)
    for (s, o, r) in thisGraph.edges.data('label'):
        if (s == subj) and (o == obj) and (r == rel):
            return True
    return False


def return_ratio_of_mapped_target_predicates(tgt):  # returnMappingRatio(sourceGraph)
    number_mapped_preds = number_unmapped_preds = 0
    lis = returnEdgesAsList(tgt)
    for (s, v, o) in lis:
        if (s in GM.mapping.keys()) and (v in GM.mapping.keys()) and (o in GM.mapping.keys()):
            number_mapped_preds += 1
        else:
            number_unmapped_preds += 1
    tmp = number_mapped_preds + number_unmapped_preds
    if tmp == 0:
        rslt = 0
    else:
        rslt = number_mapped_preds / tmp
    return number_mapped_preds, rslt


def return_ratio_of_mapped_source_predicates(tgt):  # returnMappingRatio(sourceGraph)
    number_mapped_preds = number_unmapped_preds = 0
    lis = returnEdgesAsList(tgt)
    for (s, v, o) in lis:
        if (s in GM.mapping.values()) and (v in GM.mapping.values()) and (o in GM.mapping.values()):
            number_mapped_preds += 1
        else:
            number_unmapped_preds += 1
    tmp = number_mapped_preds + number_unmapped_preds
    if tmp == 0:
        rslt = 0
    else:
        rslt = number_mapped_preds / tmp
    return number_mapped_preds, rslt


def printMappedPredicates(graf):  # printMappedPredicates(sourceGraph)
    mapped = notMapped = 0
    lis = returnEdgesAsList(graf)
    global list_of_mapped_preds
    global analogyFilewriter
    #    list_of_mapped_preds = []
    flag = False
    unmapped = set()
    for (s, v, o) in lis:
        s_mapped = s in GM.mapping.keys()
        v_mapped = v in GM.mapping.keys()
        o_mapped = o in GM.mapping.keys()
        if s_mapped or o_mapped:  # or v_mapped
            print("{: <20} {: >1} {: >10} {: >1} {: >20}".format(s, " ", v, " ", o),
                  end="  ==   ")  # Full predicate mapped
            out_list = [s, v, o, "    ==    "]
            if s_mapped:  # partial predicate mapping
                print("{: <20}".format(GM.mapping.get(s)), end=" ")
                out_list.append(GM.mapping.get(s))
            else:
                print("{: >20}".format("_"), end="")
                out_list.append(" _")
                unmapped.add(s)
            if v_mapped:
                print("{: >10}".format(GM.mapping.get(v)), end=" ")
                out_list.append(GM.mapping.get(v))
                flag = True
            else:
                print("{: >10}".format("_ "), end="")
                out_list.append(" _ ")
                unmapped.add(v)
            if o_mapped:
                print("{: >20}".format(GM.mapping.get(o)), end="")
                out_list.append(GM.mapping.get(o))
            else:
                print("{: >20}".format("_"), end="")
                out_list.append(" _ ")
                unmapped.add(o)
            if flag:
                rslt = wn_sim(v, GM.mapping.get(v), 'v')
                tmp = round((float(rslt[0]) + (float(rslt[2]))) / 2, 2)
                print("  ", tmp)
                out_list.append(tmp)
            else:
                print()
        else:
            out_list = []
            unmapped.add(s)
            unmapped.add(v)
            unmapped.add(o)
        flag = False
        if out_list:
            analogyFilewriter.writerow(out_list)
    print(" UNMAPPED: ", unmapped)
    analogyFilewriter.writerow(["UNMAPPED: ", str(unmapped).replace(",", " ")])
    return mapped, notMapped  # numeric summary


def pm():
    printMappedPredicates(sourceGraph)


def show_graph(TgtGraph):
    plt.figure()
    pos_nodes = nx.spring_layout(TgtGraph, k=0.2, pos=None, fixed=None, iterations=150,
                                 threshold=0.01, weight='weight', scale=2, center=None, dim=2, seed=None)
    nx.draw(TgtGraph, pos_nodes, node_color='gold', with_labels=False)
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.05)
    node_attrs = nx.get_node_attributes(TgtGraph, 'label')
    custom_node_attrs = {}
    for node, attr in node_attrs.items():
        custom_node_attrs[node] = attr
    edge_labels1 = dict([((u, v,), d['label'])
                         for u, v, d in TgtGraph.edges(data=True)])
    nx.draw_networkx_edge_labels(TgtGraph, pos_attrs, alpha=0.7, edge_labels=edge_labels1)
    nx.draw_networkx_labels(TgtGraph, pos_attrs, labels=custom_node_attrs)

    plt.title("abc")  # sourceGraph.graph['Graphid'])
    plt.axis('off')
    plt.show()


# show_graph(targetGraph)


# Shows graph in html/javascript D3js force directed graph.
def show_graph_in_FF(Graph):
    data = {
        'nodes': [],
        'edges': []
    }
    for index, node in enumerate(Graph.nodes()):
        # print(index, node, end="  ")
        Graph.nodes[node]['index'] = index  # some nodes may not have label attribute set
    # print("", end="")
    for node_id, attr in Graph.nodes(data=True):
        # print(node_id, attr, end="  ")
        data['nodes'].append({'label': attr['label']})

    for source, target, attr in Graph.edges(data=True):
        data['edges'].append(
            {
                'source': Graph.nodes[source]['index'],
                'target': Graph.nodes[target]['index'],
                'label': attr['label']
            }
        )
    graphName = Graph.graph['Graphid']
    if graphName.endswith('.csv'):
        graphName = graphName[:-4]
    if not os.path.exists(os.path.dirname(htmlBranch)):
        try:
            os.makedirs(os.path.dirname(htmlBranch), exist_ok=True)
        except OSError as exc:  # race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(htmlBranch + graphName + '.json', 'w+') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)
    try:
        with open('Template.html', 'r') as file:
            htmlText = file.read()
    except IOError:
        print("ERROR - Can't find the file: Template.html")
    htmlText = htmlText.replace('XXXX', graphName)  # containing "XXXX.json" data
    with open(htmlBranch + graphName + '.html', 'w+') as outHtmlFile:
        outHtmlFile.write(htmlText)
    outHtmlFile.close()
    x = webbrowser.get()
    x = 1
    #    subprocess.run([r'C:\Program Files\Mozilla Firefox\Firefox.exe',
    #                           os.path.realpath(htmlBranch + graphName + '.html')])
    webbrowser.open(os.path.realpath(htmlBranch + graphName + '.html'), new=2)


#    webbrowser.open_new_tab( os.path.realpath(htmlBranch + graphName + '.html') )


def ff():
    show_graph_in_FF(targetGraph)


def fs():
    show_graph_in_FF(sourceGraph)


def ft():
    show_graph_in_FF(targetGraph)


# ************************************************************************
# ************************* Cache and Similarity *************************
# ************************************************************************

def read_wn_cache_to_dict():
    global CachePath
    global WN_cache
    WN_cache = {}
    with open(CachePath, "r") as wn_cache_file:
        filereader = csv.reader(wn_cache_file)
        for row in filereader:
            try:
                WN_cache[row[0] + "-" + row[1]] = row[2:]
            except IndexError:
                pass
            # read_wn_cache_to_dict()


# print("WordNet Cache initialised.         ", end="")

def wn_sim(w1, w2, partoS):
    """ wn_sim("create","construct", 'v') -> [0.6139..., 'make(v.03)', 0.666..., 'make(v.03)'] """
    global LCSLlist
    lin_max = wup_max = 0
    LCSL_temp = LCSW_temp = []
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    LCSL = LCSW = "-"
    if w1 == w2:
        return [1, w1, 1, w2]
    elif w1 + "-" + w2 in WN_cache:
        zz = WN_cache[w1 + "-" + w2]
        if zz[0] == partoS:
            lin_max = zz[1]
            wup_max = zz[2]
            LCSL = zz[3]
            LCSW = zz[4]
    else:
        syns1 = wn.synsets(w1, pos=partoS)
        syns2 = wn.synsets(w2, pos=partoS)
        for ss1 in syns1:
            for ss2 in syns2:
                lin = ss1.lin_similarity(ss2, brown_ic)  # semcor_ic) #brown_ic
                wup = ss1.wup_similarity(ss2)
                if lin > lin_max:
                    lin_max = lin
                    ss1_Lin_temp = ss1
                    ss2_Lin_temp = ss2
                    LCSL_temp = ss1.lowest_common_hypernyms(ss2)
                if wup > wup_max:
                    wup_max = wup
                    ss1_Wup_temp = ss1
                    ss2_Wup_temp = ss2
                    LCSW_temp = ss1.lowest_common_hypernyms(ss2)  # may retrun []
                if lin is None:
                    lin = 0
                if wup is None:
                    wup = 0
        if lin_max > 0:
            LCSL_temp = ss1_Lin_temp.lowest_common_hypernyms(ss2_Lin_temp)
            LCSL = simplifyLCSList(LCSL_temp)
        if wup_max > 0:
            LCSW_temp = ss1_Wup_temp.lowest_common_hypernyms(ss2_Wup_temp)
            LCSW = simplifyLCSList(LCSW_temp)
        if lin_max < 0.0000000001:
            lin_max = 0
            LCSW = simplifyLCSList(LCSW_temp)
        # print(" &&&& LCSW_temp2 ", LCSW_temp)
        if LCSW_temp == []:
            LCSW = "Synset('null." + partoS + ".02')"
        write_to_wn_cache_file(w1, w2, partoS, lin_max, wup_max, LCSL, LCSW)
    LCSLlist.append(LCSL)  # for the GUI presentation
    if LCSL_temp == []:
        LCSL = "Synset('null." + partoS + ".0303')"
    if LCSW_temp == []:
        LCSW = "Synset('null." + partoS + ".0404')"
    return [lin_max, LCSL, wup_max, LCSW]


# wn_sim("create","construct", 'v')


def write_to_wn_cache_file(w1, w2, pos, Lin, Wup, L_lcs, W_lcs):
    global CachePath
    if w1 == w2:
        return
    with open(CachePath, "a+") as wn_cache_file:
        Stringtest = w1 + "," + w2 + "," + pos
        Stringtest += "," + str(Lin) + "," + str(Wup) + "," + L_lcs + "," + W_lcs + ","
        wn_cache_file.write(" \n" + Stringtest)


###########################################################################################################

def calculate2Similarities(tgt_graph, source_graph, best_analogy):  # source concept nodes
    global arrow
    global analogyFilewriter
    global list_of_mapped_preds
    global semcor_ic
    # semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    d = 0
    j = 0
    i = 0
    max_value = 0.0
    max_value2 = 0.0
    save = []
    save2 = []
    LCSL = " "
    LCSW = " "
    arrow = "<-->"
    x = 0
    y = 0
    mapped = ""
    mapped2 = ""
    global GM
    global CSVPath
    global analogyFileName
    global CachePath
    global LCSLlist
    global mappedRelations
    global unmappedRelations
    global mappedConcepts
    global unmappedConcepts
    global list_of_inferences
    mappedRelations = 0
    unmappedRelations = 0
    mappedConcepts = 0
    unmappedConcepts = 0

    global number_of_inferences
    global list_of_inferences

    conSim = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    relSim = numpy.zeros(7)
    averageNounLin = 0
    averageNounWup = 0
    averageVerbLin = 0
    averageVerbWup = 0
    number_of_inferences = len(list_of_inferences)
    anaSim = 0.0

    tgt_preds = returnEdgesAsList(tgt_graph)
    if not os.path.exists(os.path.dirname(CSVPath + analogyFileName)):
        try:
            os.makedirs(os.path.dirname(CSVPath + analogyFileName))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                #########################
                # RELATIONAL SIMILARITY #
                #########################
    with open(CSVPath + analogyFileName, 'w+') as analogyFile:
        analogyFilewriter = csv.writer(analogyFile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        analogyFilewriter.writerow(['Type', 'Word1', 'Word2', 'Lin', 'Wup', 'LCS Lin', 'LCS Wup'])
        analogyFilewriter.writerow([analogyFileName.partition("__")[0]])  # target name
        analogyFilewriter.writerow([analogyFileName.partition("__")[2]])  # source name
        for pred in tgt_preds:
            # sPrime = pred[0].split(term_separator)[0] # head word
            # vPrime = pred[1] #.rsplit(term_separator)[1]  #oPrime = pred[2].split(term_separator)[0] # head word
            if ((mode == 'code') and (pred[0] in GM.mapping.keys()) and
                    (pred[1] in GM.mapping.keys()) and (pred[2] in GM.mapping.keys())):
                mappedRelations += 1
                res, Lin_LCS, Wu_LCS = evaluateRelationalSimilarity(pred[1], GM.mapping[pred[1]])
                analogyFilewriter.writerow(['Rel', pred[1], GM.mapping[pred[1]], res[4], res[5], Lin_LCS, Wu_LCS])
                if vbse > 3:
                    print('Rel', pred[1], GM.mapping[pred[1]], res[4], res[5], Lin_LCS, Wu_LCS)
                relSim = relSim + res
            elif (mode == 'code'):
                unmappedRelations += 1
            elif (mode == 'English' and (pred[0] in GM.mapping.keys()) and
                  (pred[1] in GM.mapping.keys()) and (pred[2] in GM.mapping.keys())):
                mappedRelations += 1
                res, Lin_LCS, Wu_LCS = evaluateRelationalSimilarity(pred[1], GM.mapping[pred[1]])
                analogyFilewriter.writerow(['Rel', pred[1], GM.mapping[pred[1]], res[4], res[5], Lin_LCS, Wu_LCS])
                if vbse > 3:
                    print('Rel', pred[1], GM.mapping[pred[1]], res[4], res[5], Lin_LCS, Wu_LCS)
                relSim = relSim + res
            elif (mode == 'English'):
                if vbse > 3:
                    print("###", pred[0], pred[1], pred[2], " \=/ ")
                unmappedRelations += 1
        # print("RELATIONS:", mappedRelations, unmappedRelations, relSim)
        #########################
        # Conceptual SIMILARITY # iterate over mapping, to avoid double counting
        #########################
        # halt()
        setOfConcepts = set()
        setOfRelations = set()
        for x in tgt_preds:
            # print(x[0], end="--")
            setOfConcepts.add(x[0])
            setOfConcepts.add(x[2])
            if vbse > 3:
                print(x[0], x[2], end="   \t")
            setOfRelations.add(x[1])
        # print("\nGID", tgt_graph.graph['Graphid'], setOfConcepts)
        for key in GM.mapping:
            if key in setOfConcepts:
                mappedConcepts += 1
                res, Lin_LCS, Wu_LCS = evaluateConceptualSimilarity(key, GM.mapping[key])
                analogyFilewriter.writerow(['Con', key, GM.mapping[key], res[4], res[5], Lin_LCS, Wu_LCS])
                if vbse > 3:
                    print('Con', key, GM.mapping[key], res[4], res[5], Lin_LCS, Wu_LCS)
                conSim = conSim + res
            if key in setOfRelations:
                mappedRelations += 1
                res, Lin_LCS, Wu_LCS = evaluateRelationalSimilarity(key, GM.mapping[key])
                analogyFilewriter.writerow(['Rel', key, GM.mapping[key], res[4], res[5], Lin_LCS, Wu_LCS])
                if vbse > 3:
                    print('Rel', key, GM.mapping[key], res[4], res[5], Lin_LCS, Wu_LCS)
                relSim = relSim + res
        unmappedConcepts = len(setOfConcepts) - mappedConcepts
        # print("CONCEPTS:", mappedConcepts, unmappedConcepts, conSim)

        # GROUNDED INFERENCES
        # generateCWSGInferences(sourceGraph, targetGraph)
        for a, r, b in list_of_inferences:
            analogyFilewriter.writerow(["Inference", a, r, b])

        # Relations
        analogyFilewriter.writerow(['#Verb Lin==0', relSim[0], 'of', (mappedRelations + unmappedRelations)])
        analogyFilewriter.writerow(['#Verb WuP==0', relSim[1], 'of', (mappedRelations + unmappedRelations)])
        if mappedRelations + unmappedRelations == 0:
            avg_Lin_relational = 0
            avg_Wup_relational = 0
        else:
            avg_Lin_relational = relSim[4] / (mappedRelations + unmappedRelations)
            avg_Wup_relational = relSim[5] / (mappedRelations + unmappedRelations)
        analogyFilewriter.writerow(['Avg Verb Lin ', avg_Lin_relational])
        analogyFilewriter.writerow(['Avg Verb Wup ', avg_Wup_relational])
        analogyFilewriter.writerow(['#Verb Lin== 1', relSim[2], 'of', (mappedRelations + unmappedRelations)])
        analogyFilewriter.writerow(['#Verb WuP== 1', relSim[3], 'of', (mappedRelations + unmappedRelations)])
        # Concepts
        analogyFilewriter.writerow(['#Noun Lin==0', conSim[0], 'of', (mappedConcepts + unmappedConcepts)])
        analogyFilewriter.writerow(['#Noun WuP==0', conSim[1], 'of', (mappedConcepts + unmappedConcepts)])
        if mappedConcepts + unmappedConcepts == 0:
            avg_Lin_conceptual = 0
            avg_Wup_conceptual = 0
        else:
            avg_Lin_conceptual = conSim[4] / (mappedConcepts + unmappedConcepts)
            avg_Wup_conceptual = conSim[5] / (mappedConcepts + unmappedConcepts)
        analogyFilewriter.writerow(['Avg Noun Lin ', avg_Lin_conceptual])
        analogyFilewriter.writerow(['Avg Noun Wup ', avg_Wup_conceptual])
        analogyFilewriter.writerow(['#Noun Lin== 1', conSim[2], 'of', (mappedConcepts + unmappedConcepts)])
        analogyFilewriter.writerow(['#Noun WuP== 1', conSim[3], 'of', (mappedConcepts + unmappedConcepts)])

        average_relational_similarity = ((((avg_Lin_relational + avg_Wup_relational) / 2) * mappedRelations) * 0.5)
        #       +  ((((avgLin+avg_Wup_conceptual)/2)*mappedConcepts) * 0.5) )
        analogyFilewriter.writerow(['AnaSim=', average_relational_similarity])

        print(" ", targetGraph.graph['Graphid'].rpartition(".")[0], sourceGraph.graph['Graphid'].rpartition(".")[0],
              conSim[0], " ", conSim[1], " {:.2f}".format(avg_Lin_conceptual), " {:.2f}".format(avg_Wup_conceptual), conSim[2], " ",
              conSim[3], " ", relSim[0], " ", relSim[1], " {:.2f}".format(avg_Lin_relational), " {:.2f}".format(avg_Wup_relational),
              relSim[2], " ", relSim[3], " ", number_of_inferences, " ", mappedConcepts, " ", mappedRelations, " ",
              " {:.2f}".format(average_relational_similarity))
      #  writeSummaryFileData(targetGraph.graph['Graphid'], sourceGraph.graph['Graphid'],
      #                       conSim[0], conSim[1], avg_Lin_conceptual, avg_Wup_conceptual, conSim[2], conSim[3],
      #                       relSim[0], relSim[1], avg_Lin_relational, avg_Wup_relational, relSim[2], relSim[3],
      #                       number_of_inferences, mappedConcepts, mappedRelations, average_relational_similarity)
        #("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "#Map Conc",
        # "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "Infs", "#MapCon", "MapRels", "AnaSim")
        writeSummaryFileData(targetGraph.graph['Graphid'], sourceGraph.graph['Graphid'],
                             tgt_graph.number_of_nodes(), tgt_graph.number_of_edges(),
                             source_graph.number_of_nodes(), source_graph.number_of_edges(),
                             len(best_analogy),  (len(GM.mapping)- len(best_analogy)),
                             avg_Lin_conceptual, avg_Wup_conceptual, avg_Lin_relational, avg_Wup_relational,
                             number_of_inferences, mappedConcepts, mappedRelations, average_relational_similarity)
        printMappedPredicates(tgt_graph)
    # analogyFile.flush()
    # analogyFile.close()


################################################################################################################


def evaluateRelationalSimilarity(tRel, sRel):  # drive, walk
    global term_separator
    reslt = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    if tRel == sRel:
        reslt[2] = 1.0
        reslt[3] = 1.0
        reslt[4] = 1.0
        reslt[5] = 1.0
        LinLCS = tRel
        WuPLCS = tRel
    elif mode == 'code':
        if (tRel.split(term_separator) != []) and (
                tRel.split(term_separator)[:1] == sRel.split(term_separator)[:1]):  # Head identicality for relations?
            reslt[4] = 0.81
            reslt[5] = 0.811
            LinLCS = tRel.split(term_separator)[:1]
            WuPLCS = tRel.split(term_separator)[:1]
        elif tRel.split(term_separator)[0] == sRel.split(term_separator)[0]:
            reslt[4] = 0.33
            reslt[5] = 0.33
            LinLCS = tRel.split(term_separator)[0]
            WuPLCS = tRel.split(term_separator)[0]
        else:
            temp_result = wn_sim(tRel, sRel, 'v')
            # print("temp_result",temp_result)
            reslt[4] = temp_result[0]
            reslt[5] = temp_result[2]
            LinLCS = 'no-reln'
            WuPLCS = 'no-reln'
    else:
        temp_result = wn_sim(tRel, sRel, 'v')  # returns [0.6139, 'make(v.03)', 0.666, 'make(v.03)']
        if float(temp_result[0]) > 0 or float(temp_result[2]) > 0:
            if temp_result[0] == 0:
                reslt[0] = 1
            elif temp_result[0] == 1:
                reslt[2] = 1
            else:
                reslt[4] = temp_result[0]
            if temp_result[2] == 0:
                reslt[1] = 1
            elif temp_result[2] == 1:
                reslt[3] = 1
            else:
                reslt[5] = temp_result[2]
        LinLCS = temp_result[1]  # .find('(')
        WuPLCS = temp_result[3]
    return reslt, LinLCS, WuPLCS  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum ...


def evaluateConceptualSimilarity(tRel, sRel):  # ('fly','drive')
    global term_separator
    reslt = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    if tRel == sRel:
        reslt[2] = 1
        reslt[3] = 1
        reslt[4] = 1
        reslt[5] = 1
        LinLCS = tRel
        WuPLCS = tRel
    elif mode == 'code':
        if second_head(tRel) == second_head(sRel):
            reslt[4] = 0.95
            reslt[5] = 0.95
            LinLCS = second_head(tRel)
            WuPLCS = LinLCS
        elif tRel.split(term_separator)[:1] == sRel.split(term_separator)[:1]:
            reslt[4] = 0.8
            reslt[5] = 0.8
            LinLCS = tRel.split(term_separator)[:1]
            WuPLCS = tRel.split(term_separator)[:1]
        elif tRel.split(term_separator)[0] == sRel.split(term_separator)[0]:
            reslt[4] = 0.33
            reslt[5] = 0.33
            LinLCS = tRel.split(term_separator)[0]
            WuPLCS = tRel.split(term_separator)[0]
        else:
            reslt[0] = 0.00001
            reslt[1] = 0.00001
            LinLCS = 'none'
            WuPLCS = 'none'
    else:
        temp_result = wn_sim(tRel, sRel, 'n')  # returns [0.6139 'make(v.03)', 0.666, 'make(v.03)']
        # print("temp_result", temp_result)
        if float(temp_result[0]) > 0 or float(temp_result[2]) > 0:
            if temp_result[0] == 0:
                reslt[0] = 1
            elif temp_result[0] == 1:
                reslt[2] = 1
            else:
                reslt[4] == temp_result[0]
            if temp_result[2] == 0:
                reslt[1] = 1
            elif temp_result[2] == 1:
                reslt[3] = 1
            else:
                reslt[5] = temp_result[2]
        LinLCS = temp_result[1]
        WuPLCS = temp_result[3]
    return reslt, LinLCS, WuPLCS


########################################################################
########################################################################
########################################################################


# simplifyLCS("Synset('object.n.01')") -> "object"    simplifyLCS(["Synset('object.n.01')"]) -> "object"
# simplifyLCSList(["Synset('physical_entity.n.01')"]) -> "object"
# simplifyLCSList(["Synset('move.v.02')"] )
def simplifyLCS(synsetName):
    if isinstance(synsetName, list):
        synsetName = simplifyLCS(synsetName[0]) + simplifyLCS(synsetName[1:])
    elif (isinstance(synsetName, str)) and ("Synset" in synsetName):
        y = synsetName.find('(') + 2
        z = synsetName.find('.') + 5
        synsetName = synsetName[y:z].replace('.', '(', 1) + ")"
    elif (isinstance(synsetName, list)) and (len(synsetName) > 1):
        simplifyLCS(synsetName[0]).append(simplifyLCS(synsetName[1:]))
        simplifyLCS(str(synsetName[0])).append(simplifyLCSList(synsetName[1:]))  # 11/10
    else:  # instance of <class 'nltk.corpus.reader.wordnet.Synset'>
        ssString = str(synsetName)
        y = ssString.find('(') + 2
        z = ssString.find('.') + 5
        synsetName = ssString[y:z].replace('.', '(', 1) + ")"
    return synsetName  # [synsetName]


# simplifyLCS("[Synset('whole.n.02')]")


def simplifyLCSList(synsetList):
    if (synsetList is None):
        return "none1"
    elif (synsetList == []):
        return ""
    elif synsetList == "none":
        return "none"
    elif (isinstance(synsetList, str)):
        z = simplifyLCS(synsetList)
        return z
    elif (isinstance(synsetList, list)) and (len(synsetList) > 1):
        return str(simplifyLCS(synsetList[0])) + "_" + str(simplifyLCSList(synsetList[1:]))
    elif (isinstance(synsetList, list)) and (len(synsetList) == 1):
        return simplifyLCS(synsetList[0])
    else:
        print(" sLCSL5", end="")
        zz = simplifyLCS(synsetList)
        return zz


########################################################################
#############################   Graph   ################################
#############################  Matching   ##############################
########################################################################

def encode_graph_labels(grf):
    nu_grf = nx.OrderedGraph()
    s_encoding = {}  # label, number
    s_decoding = {}  # number, label
    label = 0
    for x, y in grf.edges():
        if not x in s_encoding.keys():
            s_decoding[label] = x
            s_encoding[x] = label
            label += 1
        if not y in s_encoding.keys():
            s_decoding[label] = y
            s_encoding[y] = label
            label += 1
        nu_grf.add_edge(s_encoding[x], s_encoding[y])
    return nu_grf, s_encoding, s_decoding


def mappingProcess(target_graph, source_graph):
    global GM
    global relationMapping
    global numberOfTimeOuts
    global list_of_mapped_preds  # share results
    global mapping_run_time
    GM.mapping.clear()
    isomorphvf2CB.temp_sol = []
    before_seconds = time.time()
    if algorithm == "DFS":
        list_of_mapped_preds, relato_struct_sim = DFS.generate_and_explore_mapping_space(sourceGraph, targetGraph)
        GM.mapping = {}
        #p, q, r = list_of_mapped_preds[0]
        for p, q, sim in list_of_mapped_preds:# [0]:  # read back the results
            a, b, c = q
            x, y, z = p
            GM.mapping[a] = x
            GM.mapping[b] = y
            GM.mapping[c] = z
        mapping_run_time = time.time() - before_seconds
        print(" DFS Time:", mapping_run_time, end="  ")
    elif algorithm == "ismags":
        s_grf, s_encoding, s_decoding = encode_graph_labels(sourceGraph)
        t_grf, t_encoding, t_decoding = encode_graph_labels(targetGraph)
        # print()         # {some_node_here: 1,  someother_node: 2 ...}
        ismags = nx.isomorphism.ISMAGS(s_grf, t_grf)
        largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))  # False
        GM.mapping = largest_common_subgraph[0].copy()
        mapping_run_time = time.time() - before_seconds
        print(" ISMAGS Time:", mapping_run_time, end="  ")
        return largest_common_subgraph, s_decoding, t_encoding
    else:
        # timeLimit = 5.0
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        GM = isomorphvf2CB.MultiDiGraphMatcher(target_graph, source_graph)
        # p1 = Process(target=isomorphvf2CB.MultiDiGraphMatcher,
        #             args=(source_graph, target_graph), name='MultiDiGraphMatcher')
        # p1.start()
        # p1.join(timeout=timeLimit)
        # p1.terminate()
        # if p1.exitcode is None:     # a TimeOut
        #   numberOfTimeOuts += 1
        #   return 0
        # p2 = Process(target=isomorphvf2CB.MultiDiGraphMatcher.subgraph_is_isomorphic,
        #             args=(GM), name='subgraph_is_isomorphic')
        # p2.start()
        # p2.join(timeout=timeLimit)
        # p2.terminate()
        # if p2.exitcode is None:     # a TimeOut
        #   numberOfTimeOuts += 1
        #   return 0
        res = GM.subgraph_is_isomorphic()
    if len(GM.mapping) == 0:
        print(":-( NO Mapping:")
    else:
        print("   ", len(GM.mapping), "mapped concepts.", 100 * len(GM.mapping) / target_graph.number_of_edges(),
              " %T  ", 100 * len(GM.mapping) / source_graph.number_of_nodes(), " %S")


def develop_analogy(target_graph, source_graph):
    global GM
    global relationMapping
    global numberOfTimeOuts
    global list_of_mapped_preds
    GM.mapping.clear()
    isomorphvf2CB.temp_sol = []
    if source_graph.number_of_nodes() == 0:
        return 0
    if algorithm == "ismags":
        list_of_dictionaries, s_decoding, t_encoding = mappingProcess(target_graph, source_graph)
        # remove_duplicate_dictionaries(list_of_dictionaries)
        interpretations = []
        print("2 ", len(list_of_dictionaries), " interpretations.  ", end="")
        for dic in list_of_dictionaries:  # encode target but decode source
            res = addRelationsToMapping(target_graph, source_graph, dic, s_decoding, t_encoding)
            interpretations += [res]
        best_analogy = sorted(interpretations, key=lambda val: val[0], reverse=True)[0][1:]
        if best_analogy == []:
            global list_of_inferences
            list_of_inferences = []
        elif best_analogy[0] == 0:
            print("** Useless Mapping ** ", end="")
        else:
            for a, b, c, q, r, s in best_analogy[1:]:
                GM.mapping[a] = q
                GM.mapping[b] = r
                GM.mapping[c] = s
    else:
        best_analogy = mappingProcess(target_graph, source_graph)
    print("", end="")
    if not algorithm == "ismags":
        best_analogy = slightly_flatten(list_of_mapped_preds)
    generateCWSGInferences(target_graph, source_graph, best_analogy)
    calculate2Similarities(target_graph, source_graph, best_analogy)
    return 0


def slightly_flatten(preds_list):
    if preds_list == []:
        return []
    else:
        rslt = []
        for entry in preds_list:
            rslt.append(entry[0] + entry[1])
    return rslt


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


def addRelationsToMapping(target_graph, source_graph, mapping_dict, s_decoding, t_decoding):  # LCS_Number
    "new one. GM.mapping={(t,s), (t2,s2)...}"
    this_mapping = mapping_dict.copy()  # was GM.mapping.
    mappedFullPredicates = []
    rel_sim_total = 0
    rel_sim_count = 0
    all_target_edges = returnEdgesAsList(target_graph)
    source_edge_list = returnEdgesAsList(source_graph)
    for tNoun1, tRelation, tNoun2 in all_target_edges:
        tNoun1_num = t_decoding[tNoun1]  # decoding ={label, number }
        tNoun2_num = t_decoding[tNoun2]
        if tNoun1_num in this_mapping.keys() and tNoun2_num in this_mapping.keys():  # and \
            #    this_mapping[tNoun1_num] in this_mapping.values() and this_mapping[tNoun2_num] in this_mapping.values():
            sNoun1_num = this_mapping[tNoun1_num]
            sNoun2_num = this_mapping[tNoun2_num]
            if not (sNoun1_num in s_decoding.keys() and sNoun2_num in s_decoding.keys()):
                break
            sNoun1 = s_decoding[sNoun1_num]  # what if null-  test require
            sNoun2 = s_decoding[sNoun2_num]
            source_verbs = returnEdgesBetweenTheseObjects_predList(sNoun1, sNoun2, source_edge_list)
            nearest_dist = 0.0
            nearest_s_verb = "nULl"
            dist = 0
            for s_verb in source_verbs:
                if s_verb:
                    if tRelation == s_verb:
                        dist = 1
                    elif s2v.get_freq(head_word(tRelation) + '|VERB') is not None and \
                            s2v.get_freq(head_word(s_verb) + '|VERB') is not None:
                        dist = s2v.similarity([head_word(tRelation) + '|VERB'], [head_word(s_verb) + '|VERB'])
                    else:
                        print("**~ S2V", tRelation, s_verb, end="*   ")
                if dist >= nearest_dist:
                    nearest_dist = dist
                    nearest_s_verb = s_verb
                    cached_predicate_mapping = [[tNoun1, tRelation, tNoun2, sNoun1, nearest_s_verb, sNoun2]]
            if not nearest_s_verb == 'nULl':
                rel_sim_total += nearest_dist
                rel_sim_count += 1
                this_mapping[tRelation] = nearest_s_verb
                mappedFullPredicates += cached_predicate_mapping
                source_edge_list.remove([sNoun1, nearest_s_verb, sNoun2])
    mappedFullPredicates.insert(0, rel_sim_total)
    return mappedFullPredicates  # rel_sim, rel_sim_count,


def returnMappedConcept(s_con):
    global GM
    global term_separator
    if mode == 'code':
        if s_con in GM.mapping.keys():
            return GM.mapping[s_con]
        else:
            return False
    else:
        for thingy in GM.mapping.keys():
            if s_con in thingy.split(term_separator)[0]:
                return GM.mapping[thingy]
        return False


#####################
# Open output files
#####################

def generateCWSGInferences(tgtGraph, srcGrf, mapped_preds_lis):  # generateCWSGInferences(sourceGraph, targetGraph)
    global GM
    global list_of_inferences
    if mapped_preds_lis == []:
        return
    srcEdges = returnEdgesAsList(srcGrf)
    for a, b, c, d, e, f in mapped_preds_lis:  # eliminate mapped predicates
        for h, i, j in srcEdges:
            if a == h and b == i and c == j:
                srcEdges.remove([h, i, j])
                break
    for subj, reln, obj in srcEdges:
        s = r = o = 0
        if subj in GM.mapping:
            s = 1
        if obj in GM.mapping:
            o = 1
        if reln in GM.mapping or reln in ['assert']:  # Aris
            r = 1
        if reln in ['assert'] and mode == 'code':
            print(" ASSERT found ", end="")
        subjMap = GM.mapping.get(subj)
        relnMap = GM.mapping.get(reln)
        objMap = GM.mapping.get(obj)
        if not predExists(subjMap, relnMap, objMap, tgtGraph):  # generate inference
            if s + r + o == 2:
                if subjMap == None:  # Generate transferrable symbol
                    subjMap = subj
                if objMap == None:
                    objMap = obj
                if relnMap == None:
                    relnMap = reln
                print(" #INFER(", subj, ", ", reln, ", ", obj, " => ", subjMap, ", ", relnMap, ", ", objMap, end=")   ")
                list_of_inferences = list_of_inferences + [[subjMap, relnMap, objMap]]
    print("\n", end="")


def writeSummaryFileData(fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q):  # 18 params
    global CSVsummaryFileName
    global mapping_run_time
    with open(CSVsummaryFileName, "a+") as csvSummaryFileHandle:
        summaryFilewriter = csv.writer(csvSummaryFileHandle, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        summaryFilewriter.writerow(
            [fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q, mapping_run_time])
#("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "#Map Conc",
# "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "Infs", "#MapCon", "MapRels", "AnaSim")
def writeSummaryFileData(fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o):  # 18 params
    global CSVsummaryFileName
    global mapping_run_time
    with open(CSVsummaryFileName, "a+") as csvSummaryFileHandle:
        summaryFilewriter = csv.writer(csvSummaryFileHandle, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        summaryFilewriter.writerow(
                    [fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, mapping_run_time])


################################################################
################ Run Multiple Analogies ########################
################################################################

def resetAnalogyMetrics():
    global inference
    inference = 0
    global list_of_inferences
    list_of_inferences = []
    global LCSLlist
    LCSLlist = []
    global list_of_mapped_preds
    list_of_mapped_preds = []
    global GM
    GM = isomorphvf2CB.MultiDiGraphMatcher(targetGraph, targetGraph)
    # GM = isomorphvf2CB.DiGraphMatcher(targetGraph, targetGraph)
    # global sourceGraph
    global relationMapping
    relationMapping = []
    global numberOfTimeOuts
    numberOfTimeOuts = 0


def blendWithAllSources(targetFile):  # blendAllSources(targetFile)
    global all_csv_files  # supply target, look for sources
    global targetGraph
    global sourceGraph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    if False:   # file head restriction for experimental groups
        targetFileHead = targetFile[0:5]             ## similar file name restriction
        these_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith(targetFileHead)]
        print("\n these_csv_files", targetFileHead, " ", these_csv_files, end=" with ")
        all_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) ]
        print(len(these_csv_files), " candidate sources.")
    targetGraph = build_graph_from_csv(targetFile).copy()
    targetGraph.graph['Graphid'] = targetFile
    p1 = targetFile.rfind(".")  # filetypeFilter
    if targetGraph.number_of_edges() > max_graph_size:
        prune_peripheral_nodes(targetGraph)
    for nextSourceFile in all_csv_files:
        # if nextSourceFile == targetFile: # skip self comparison
        #    continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile, "  <- ", nextSourceFile[0:p2], "=======")
        analogyFileName = targetFile[0:p1] + "__" + nextSourceFile[0:p2] + ".csv"
        verbotenFile = targetFile[0:p1] + "-" + nextSourceFile[0:p2] + ".csv"
        if path.isfile(CSVPath + analogyFileName):
            print(' £££skippy  ', end="")
            continue
        if verbotenFile in ['Iso-10.cs-Iso-27.cs.csv', 'Iso-19.cs-Iso-11.cs.csv', 'Iso-11.cs-diss-10.cs.csv',
                            'Iso-12.cs-diss-10.cs.csv', 'Iso-19.cs-Iso-23.cs.csv', 'Iso-19.cs-Iso-24.cs.csv',
                            'near-iso-8.cs-near-iso-33.cs.csv', 'near-iso-8.cs-homo-33.cs.csv',
                            'near-iso-8.cs-diss-33.cs.csv', 'near-iso-8.cs-Iso-33.cs.csv',
                            'Iso-19.cs-Iso-27.cs.csv', 'Iso-19.cs-Iso-28.cs.csv', 'Iso-19.cs-Iso-29.cs.csv',
                            'near-iso-7.cs-near-iso-42.cs.csv', 'diss-10.cs-diss-33.cs.csv',
                            'near-iso-7.cs-near-iso-35.cs.csv', 'near-iso-7.cs-near-iso-33.cs.csv',
                            'Iso-19.cs-Iso-32.cs.csv', 'Iso-19.cs-Iso-33.cs.csv', 'near-iso-7.cs-homo-42.cs.csv',
                            'Iso-19.cs-Iso-35.cs.csv', 'Iso-19.cs-Iso-37.cs.csv', 'near-iso-7.cs-homo-35.cs.csv',
                            'near-iso-7.cs-homo-33.cs.csv', 'Iso-19.cs-Iso-38.cs.csv', 'near-iso-7.cs-diss-42.cs.csv',
                            'near-iso-7.cs-diss-35.cs.csv', 'Iso-19.cs-Iso-40.cs.csv', 'Iso-19.cs-Iso-41.cs.csv',
                            'near-iso-7.cs-diss-35.cs.csv', 'near-iso-7.cs-diss-33.cs.csv', 'Iso-19.cs-Iso-42.cs.csv',
                            'Iso-19.cs-Iso-43.cs.csv', 'Iso-19.cs-Iso-44.cs.csv', 'near-iso-7.cs-Iso-35.cs.csv',
                            'Iso-19.cs-Iso-47.cs.csv', 'Iso-19.cs-Iso-5.cs.csv', 'near-iso-7.cs-Iso-33.cs.csv',
                            'near-iso-6.cs-near-iso-42.cs.csv', 'near-iso-6.cs-near-iso-33.cs.csv',
        'Iso-21.cs-Iso-27.cs.csv', 'near-iso-6.cs-near-iso-27.cs.csv','Iso-21.cs-Iso-42.cs.csv',
                'near-iso-6.cs-homo-42.cs.csv', 'near-iso-6.cs-homo-33.cs.csv', 'Iso-21.cs-diss-27.cs.csv',
            'Iso-21.cs-diss-42.cs.csv', 'near-iso-6.cs-homo-27.cs.csv', 'Iso-21.cs-homo-27.cs.csv',
            'near-iso-6.cs-diss-42.cs.csv']:  #ISMAGS skip timeout pairs
            print('verboten ', end="")
            continue
        elif targetFile in ['Iso-19.cs.csv', 'Iso-21.cs.csv']:
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        #nextSourceFile = "g0pB_taska.txt.corf.csv"
        sourceGraph = build_graph_from_csv(nextSourceFile).copy()
        sourceGraph.graph['Graphid'] = nextSourceFile
        if sourceGraph.number_of_edges() > max_graph_size:
            prune_peripheral_nodes(sourceGraph)
        # show_graph_in_FF(sourceGraph)
        develop_analogy(targetGraph, sourceGraph)
        #develop_analogy(sourceGraph, targetGraph)
        print("Source map:", return_ratio_of_mapped_source_predicates(sourceGraph),  # ...Source_Predicates
              "\tTarget map:", return_ratio_of_mapped_target_predicates(targetGraph))
        # if analogyCounter >= run_Limit:
        #    sys.exit()
        #stop()
        analogyCounter += 1  # analogyFileName.close()
        # anyKey = input()


def blendToPossibleTargets(sourceFile):  # targetFile):  # blendAllSources(targetFile)
    global all_csv_files  # supply source, look for targets
    global targetGraph
    global sourceGraph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    print("\nExploring Source::", sourceFile, end="")
    sourceGraph = build_graph_from_csv(sourceFile).copy()
    sourceGraph.graph['Graphid'] = sourceFile
    p1 = sourceFile.rfind(".")  # filetypeFilter
    if sourceGraph.number_of_edges() > max_graph_size:
        prune_peripheral_nodes(sourceGraph)
    # csv_sources = ['S1.csv', 'S2.csv', 'S3.csv']
    for nextTargetFile in all_csv_files:
        # if nextTargetFile == sourceFile: # skip self comparison
        #    continue
        p2 = nextTargetFile.rfind(".")
        print("\n=====zzz=============", end=" ")
        print("#", mode, "   ", nextTargetFile[0:p2], "  <- ", sourceFile[0:p1], "=======")
        print("#==========", mode, "   ", sourceFile, "  <- ", nextTargetFile[0:p2], "=======")
        analogyFileName = nextTargetFile[0:p2] + "__" + sourceFile[0:p1] + ".csv"
        resetAnalogyMetrics()
        temp_graph2.clear()
        targetGraph = build_graph_from_csv(nextTargetFile).copy()
        targetGraph.graph['Graphid'] = nextTargetFile
        if targetGraph.number_of_edges() > max_graph_size:
            # continue
            prune_peripheral_nodes(targetGraph)
        show_graph_in_FF(sourceGraph)
        # DFS.mappingTest(targetGraph, sourceGraph)
        develop_analogy(targetGraph, sourceGraph)
        print("Target map:", return_ratio_of_mapped_target_predicates(targetGraph),
              "\tSource map:", return_ratio_of_mapped_target_predicates(sourceGraph))
        if analogyCounter >= run_Limit:
            sys.exit("Analogy Limit Reached.")
        analogyCounter += 1  # analogyFileName.close()
        # anyKey = input()


def blendAllFiles():
    global all_csv_files
    global analogyCounter
    global CSVPath
    global GM
    global csvSumryFil
    analogyCounter = 0
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    #writeSummaryFileData("TARGET", "SOURCE", "#Lin=0", "#WuP=0", "LinAvg", "WuPAvg", "#Lin=1", "WuP=1",
    #                     "#Lin=0", "#WuP=0", "LinAvg", "WuPAvg", "#Lin=1", "WuP=1", "Infs", "#MapCon", "MapRels", "AnaSim")
    writeSummaryFileData("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "#Map Conc",
                    "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "Infs", "#MapCon", "MapRels", "AnaSim, mS")
    # targetGraph = build_graph_from_csv(all_csv_files[0])
    # GM = isomorphvf2CB.MultiDiGraphMatcher(targetGraph, targetGraph)
    # GM.is_isomorphic()
    # temp_csv_files = ['S5-ResizeDemoAdd.csv']

    temp_csv_files = all_csv_files
    #temp_csv_files.remove('diss-10.cs.csv')
    for next_target_file in temp_csv_files:
    #for next_target_file in all_csv_files:
        print("\nNew Target: ", next_target_file, "------------------------------------------------------", end=" ")
        targetGraph = build_graph_from_csv(next_target_file)
        targetGraph.graph['Graphid'] = next_target_file
        if False:
            show_graph_in_FF(targetGraph)
             # threeWordSummary(targetGraph)
        blendWithAllSources(next_target_file)  # next_target_file #one_csv_file[0]
        # blendToPossibleTargets(next_target_file) # treat "target" as as a source
        analogyCounter += 1
    # csvSumryFil.close()
    # print(invalid_graphs, "of ",len(v)," Invalid graphs.  ","\n", analogyCounter, "analogies explored")


def threeWordSummary(Grf):
    simplifiedGraf = nx.Graph(Grf)  # simplify to NON-Multi - Graph
    from networkx.algorithms import tree
    mst = tree.minimum_spanning_edges(simplifiedGraf, algorithm="kruskal", data=False)
    edgelist = list(mst)
    pr = nx.pagerank(simplifiedGraf, alpha=0.8)  # PageRank  WEIGHT = Count Edges
    predsList = returnEdgesAsList(Grf)
    predsList2 = []
    for (n1, n2) in edgelist:
        for [n3, r, n4] in predsList:
            if (n1 == n3 and n2 == n4) or (n1 == n4 and n2 == n3):
                scor = pr[n1] * pr[n2]
                predsList2.append([scor, n1, r, n2])
                break
    predsList2.sort(key=lambda x: x[0], reverse=True)
    print()
    for [_, n1, r, n2] in predsList2[0:5]:
        print(n1, r, n2, end="   ")
    print()


def prune_peripheral_nodes(grf):
    global max_graph_size
    cntr = 0
    print("\nPruning nodes from ", grf.size(), end="-> ")
    limit = grf.number_of_edges() - max_graph_size
    while grf.number_of_edges() > max_graph_size:
        degree_sequence = sorted([(d, n) for n, d in grf.degree()])
        for degr, lab in degree_sequence:
            print("del ", end="")
            grf.remove_node(lab)  # possibly Many edges deleted
            cntr += 1
            if (cntr >= limit) or (cntr % 5 == 0):  # delete nodes in batches of 5
                break
    print(grf.number_of_edges(), end="   ")


#######################################################################
###########################   GUI    ##################################
#######################################################################

def exitApp():
    window.destroy()
    sys.exit()


def retrieve_input():
    print("getting input")
    targetText4 = Entry(window)
    # targetText4.pack()
    targetText4.delete(0, END)
    targetText4.insert(INSERT, "Type text here")
    # sourceText5.delete(0, END)
    inputValue1 = targetText4.get("1.0", "end-1c")
    # inputValue2=sourceText5.get("1.0","end-1c")
    # targetText4.set("Mary had a little lamb\nHer Fleece was white as snow.\n")
    # print("inputValue", inputValue)
    # execfile('Text2ROS.py')
    Text2Predic8.processAllTextFiles()
    Text2Predic8.processDocument(inputValue1)
    Text2Predic8.generate_output_CSV_file("Antonet01 - Four-Canals-Carocci.txt.csv")
    Text2Predic8.processAllTextFiles()
    Text2Predic8.processDocument(inputValue2)
    Text2Predic8.generate_output_CSV_file("Antonet01 - Lake-Complete-Source-Carocci.txt.csv")
    # targetText4.insert(INSERT, list(sourceGraph.nodes))
    # sourceText5.insert(INSERT, list(targetGraph.nodes))
    return


def ok():
    global sourceGraph
    global targetGraph
    with open(CSVPath + CSVName, 'w+') as analogyCsvfile:
        filewriter = csv.writer(analogyCsvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow(['Type', 'Noun1', 'Noun2', 'Lin', 'Wup', 'LCS Lin', 'LCS Wup'])
    p1 = sourceGraph.graph['Graphid'].find(".")
    p2 = targetGraph.graph['Graphid'].find(".")
    print("####################################")
    print("### ", sourceGraph.graph['Graphid'][0:p1], "  <->", targetGraph.graph['Graphid'][0:p2], " ##")
    print("####################################")

    printEdges(sourceGraph)
    targetText4.insert(INSERT, returnEdges(sourceGraph))  # list(sourceGraph.nodes)
    sourceText5.insert(INSERT, returnEdges(targetGraph))
    develop_analogy(targetGraph, sourceGraph)
    global LCSLlist
    global list_of_mapped_preds
    # addRelationsToMapping(sourceGraph, targetGraph)
    CSV()
    analogyCsvfile.close()
    lcsText1.insert(INSERT, LCSLlist)
    inferencesText2.insert(INSERT, list_of_inferences)
    mappingText3.insert(INSERT, list_of_mapped_preds)


def temp():
    window = tkinter.Tk()
    window.title("Cre8blend - GUI")
    lcsText1 = Text(window, height=15, width=30)
    targetLabel = Label(window, text="Input 1", bg="white", fg="black")  # .grid(row=0, sticky=E)
    sourceLabel = Label(window, text="Input 2", bg="white", fg="black")
    L = Label(window, text="Shared - Lowest Common Subsumer", bg="white", fg="black")
    a = Button(window, text="Run Single Analogy", command=ok)  ###
    # b = Button(window, text = "Map ALL sources",command = ok)  ###
    w = Label(window, text="Mapping", bg="white", fg="black")
    mappingText3 = Text(window, height=15, width=35)
    a.pack(expand=True)
    L.pack(expand=True)
    lcsText1.pack(side=TOP)
    scrollbar = Scrollbar(window)
    scrollbar.pack(side=LEFT, fill=Y)
    scrollbar2 = Scrollbar(window)
    scrollbar2.pack(side=RIGHT, fill=Y)
    targetText4 = Text(window, height=20, width=40)
    sourceText5 = Text(window, height=25, width=40)
    targetLabel.pack(side=LEFT, expand=True)
    targetText4.pack(side=LEFT, expand=True)
    sourceLabel.pack(side=RIGHT, expand=True)
    sourceText5.pack(side=RIGHT, expand=True)
    scrollbar.config(command=targetText4.yview)
    scrollbar2.config(command=sourceText5.yview)
    buttonCommit = Button(window, height=1, width=10, text="Make Graph",  ### input one
                          command=lambda: retrieve_input())
    buttonCommit.pack()
    targetText4.config(yscrollcommand=scrollbar.set)
    sourceText5.config(yscrollcommand=scrollbar2.set)
    w.pack()
    mappingText3.pack(expand=True)
    l = Label(window, text="Inference", bg="white", fg="black")
    l.pack(expand=True)
    inferencesText2 = Text(window, height=15, width=20)
    inferencesText2.pack(side=BOTTOM, expand=True)

    # Button(window, text="Quit", bg="white", fg="black", command=exitApp).pack()
    Button(window, text="Quit", bg="white", fg="black", command=exitApp).pack()
    # myString = StringVar()
    # entry1 = Entry(window, textVariable = myString)
    window.mainloop()


# CN Elaboration, CN_file -> CN_dict

def loadConceptNet(CNfileNam):
    # csvreader = csv.reader(CNfileNam, delimiter=',', quotechar='|')
    reader = csv.reader(open(CNfileNam, 'r'))  # , errors='replace')
    for row in reader:
        noun1, noun2, verb = row
        try:
            CN_dict[noun1].append((verb, noun2))
        except KeyError:
            CN_dict[noun1] = [(verb, noun2)]


# loadConceptNet(CNfileNam)

def checkCNConnection(noun1, noun2):
    # Returns true if CN_dictionary has noun1 as key & value (any verb, noun2)
    r = False
    try:
        for pair in CN_dict[noun1]:
            if not r:
                r = noun2 in pair[1]
    except KeyError:
        r = False
    return r


# checkCNConnection('four','4')   checkCNConnection('four','rowing')

def getCNConnections(noun1, noun2):
    # Returns a list of verbs that connect noun1 and noun2 in the ConceptNet data
    connections = []
    for pair in CN_dict[noun1]:
        if noun2 == pair[1]:
            connections.append(pair[0])
    return connections


# getCNConnections('four','4')


def getNodes(graph):
    tempList = []
    for node in Graph_dictionary:
        tempList.append(node)
        for pair in graph[node]:
            tempList.append(pair[1])
    nodes = list(set(tempList))
    return nodes


# getNodes(sourceGraph)


def findMissingConnections(graph):
    # Takes pairs of nodes and uses the checkCNConnection to see if a connection should be added to the graph
    # nodes = getNodes(graph)
    nodes = sourceGraph.nodes()
    # list(graph) is used here to avoid runtime error: dictionary changed size during iteration
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                if checkCNConnection(n1, n2):
                    for verb in getCNConnections(n1, n2):
                        addConnection(n1, n2, verb)
                if checkCNConnection(n2, n1):
                    for verb in getCNConnections(n2, n1):
                        addConnection(n2, n1, verb)
                        # Checks connection both directions as graph is directed


def addConnection(noun1, noun2, verb):
    # Adds the connection noun1 -> verb -> noun2
    if noun1 in Graph_dictionary:
        if (verb, noun2) not in Graph_dictionary[noun1]:
            Graph_dictionary[noun1].append((verb, noun2))
            print("added-a " + noun1 + " " + verb + " " + noun2)
    else:
        Graph_dictionary[noun1] = [(verb, noun2)]
        print("added-b " + noun1 + " " + verb + " " + noun2)
        # Prints the keys/values of the newly added connection


def findNewNodes(graph):
    # Creates a list containing all the current nodes in our graph
    nodes = getNodes(graph)
    # Iterates over groups of three nodes and checks if they share a common connected node using shareCommonNode
    # Change this to four nodes if you wish to increase the requirement
    BannedEdges = []
    for n1 in nodes:
        for n2 in nodes:
            for n3 in nodes:
                if n1 != n2 and n2 != n3 and n1 != n3:
                    toAdd0 = shareCommonNode(n1, n2, n3)
                    if toAdd0 != []:
                        # If all three nodes share a common connected node
                        # That node is added along with the relevant connection
                        for node in toAdd0:
                            for verb in getCNConnections(n1, node):
                                if verb not in BannedEdges:
                                    addConnection(n1, node, verb)
                            for verb in getCNConnections(n2, node):
                                if verb not in BannedEdges:
                                    addConnection(n2, node, verb)
                            for verb in getCNConnections(n3, node):
                                if verb not in BannedEdges:
                                    addConnection(n3, node, verb)
    print('')
    for key in CN_dictionary:
        toAdd1 = []
        for pair in CN_dictionary[key]:
            if pair[1] in nodes:
                toAdd1.append(pair[1])
        if len(toAdd1) >= 3:
            # This enforces the restriction of three current nodes, edit if you wish to change the requirement
            for node in toAdd1:
                for verb in getCNConnections(key, node):
                    if verb not in BannedEdges:
                        addConnection(key, node, verb)


def t():
    findMissingConnections(sourceGraph)


#######################################################################################################

def call_DFS():
    # import DFS
    print()
    nextTargetFile1 = all_csv_files[0]
    temp_graph1 = nx.MultiDiGraph()
    temp_graph1 = build_graph_from_csv(nextTargetFile1)
    nextTargetFile2 = all_csv_files[3]
    print("\nall_csv_files[3]", all_csv_files[3])
    temp_graph2 = nx.MultiDiGraph()
    temp_graph2 = build_graph_from_csv(nextTargetFile2)
    DFS.generate_and_explore_mapping_space(temp_graph1, temp_graph2)
    print("================End Mapping Test=================")


# import DFS

def testDFS():
    global all_csv_files
    global analogyCounter
    global CSVPath
    analogyCounter = 1
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    writeSummaryFileData("TARGET", "SOURCE", "#Lin=0", "#WuP=0", "LinAvg", "WuPAvg", "#Lin=1", "WuP=1",
                         "#Lin=0", "#WuP=0", "LinAvg", "WuPAvg", "#Lin=1", "WuP=1",
                         "Infs", "#MapCon", "MapRels", "AnaSim")
    sourceGraphList = ['1054463.cs.csv']  # ['S2.csv'] #, 'S2.csv','S3.csv']
    sourceGraph = build_graph_from_csv(sourceGraphList[0])
    for nextTarget in all_csv_files:
        print("\n================", nextTarget, "================ ")
        targetGraph = build_graph_from_csv(nextTarget)  # for Display-only usage
        targetGraph.graph['Graphid'] = nextTarget
        # show_graph_in_FF(targetGraph)
        # print("XXXX", targetGraph.nodes(), sourceGraph.nodes())
        # show_graph_in_FF(sourceGraph)
        rslt = DFS.generate_and_explore_mapping_space(targetGraph, sourceGraph)  # Deterministic Depth First Search DFS
        rslt = DFS.generate_and_explore_mapping_space(sourceGraph, targetGraph)  # Deterministic Depth First Search DFS
        print()
    print("\n", analogyCounter, "analogies explored")


# GM = isomorphvf2CB.MultiDiGraphMatcher(targetGraph, targetGraph)
# GM = isomorphvf2CB.GraphMatcher(ztemp2OrdinaryGraph, ztemp2OrdinaryGraph)
# GM.is_isomorphic()


def quick_wasserstein(a, b):  # inspired by Wasserstein distance (quick Wasserstein approximation)
    a_prime = sorted(list(set(a) - set(b)))
    b_prime = sorted(list(set(b) - set(a)))
    if len(a_prime) < len(b_prime):  # longer list is the prime
        temp = b_prime.copy()
        b_prime = a_prime.copy()
        a_prime = temp.copy()
    sum1 = sum(abs(i - j) for i, j in zip(a_prime, b_prime))
    b_len = len(b_prime)
    sum2 = sum(a_prime[b_len:])
    return sum1 + sum2


assert quick_wasserstein([1, 2, 3], [1, 2, 3]) == 0
assert quick_wasserstein([1, 2, 3], [1, 2, 4, 5]) == 6
assert quick_wasserstein([0, 1, 3], [5, 6, 8]) == 15


# print("\ndifference([1,2,3],[1,2, 4, 5])", quick_wasserstein([1, 2, 3], [1, 2, 4, 5]))

def return_identical_predicates(graph1, graph2):
    lis1 = returnEdgesAsList(graph1)
    lis2 = returnEdgesAsList(graph2)


blendAllFiles()
