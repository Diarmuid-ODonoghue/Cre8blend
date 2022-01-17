import networkx as nx
import isomorphvf2CB  # VF2 My personal Modified VF2 variant
import csv
import pprint
import numpy  # as np
import matplotlib.pyplot as plt
import os   # import os.path
from os import path
import errno
import webbrowser
import json
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
import DFS
# from loguru import logger
# import snoop
# import heartrate
# from itertools import count, product
import multiprocessing   # from multiprocessing import Process, Queue, Manager
# import sys
# import ConceptNetElaboration as CN
# import pylab
# import operator
# import subprocess
# from cacheout import Cache
# import requests
# import pyvis

if True:  # supports quick testing
    from sense2vec import Sense2Vec
    # s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
    # query = "drive|VERB"
    # assert query in s2v

# ######################################################
# ############## Global Variables ######################
# ######################################################

# nx.OrderedMultiDiGraph()  #loosely-ordered, mostly-directed, somewhat-multiGraph, self-loops, parallel edges
basePath = "C:/Users/user/Documents/Python-Me/data"
basePath = "C:/Users/dodonoghue/Documents/Python-Me/data"
# basePath = dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')
global localBranch
global invalid_graphs
global max_graph_size
global algorithm
global GM
global result_graph

GM = dict()
algorithm = "DFS"
# algorithm = "ismags"
# algorithm = "VF2"

run_Limit = 50000  # 5 # Stop after this many analogies
skip_prepositions = False  # False
filetypeFilter = '.csv'  # 'txt.S.csv'
skip_over_previous_results = True  # redo, repeat,

if False:
    mode = 'code'
    max_graph_size = 500   # see prune_peripheral_nodes(graph) pruning
    term_separator = ":"  # "_"  hawk_he OR Block:Else: If
    localBranch = "/c-sharp-id-num/"
    # localBranch = "/AllCSVs/"
    # localBranch = "/C-Sharp Data/"
    # localBranch = "/Java Data/"
    # localBranch = "/TS1-TS4/"
    localBranch = "/test/"
    #localBranch = "/Java Poonam Graphs/"
else:
    mode = 'English'
    max_graph_size = 100    # see prune_peripheral_nodes(graph) pruning
    term_separator = "_"  # "_"  hawk_he
    skip_prepositions = False
    localBranch = "/test/"  # "test/" #Text2ROS.localBranch #""
    # localBranch = "/iProva/"
    # localBranch = "/Covid-19/"
    localBranch = "/Psychology data/"
    # localBranch = "/MisTranslation Data/"
    # localBranch = "/Killians Summaries/"
    # localBranch = "/SIGGRAPH ROS - Dr Inventor/"
    # localBranch = "/Sheffield-Plagiarism-Corpus/"
    # localBranch = "/20 SIGGRAPH Abstracts - Stanford/"
    # localBranch = "/SIGGRAPH csv - Dr Inventor/"

global coalescing_completed
global list_of_mapped_preds
global mapping_run_time

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

# #######################################################
# ############# File Infrastructure #####################
# #######################################################


localPath = basePath + localBranch
htmlBranch = basePath + localBranch + "FDG/"

CN_file = basePath + "/ConceptNetdata.csv"

CSVPath = localPath + "CSV Output/"  # Where you want the CSV file to be produced
CachePath = basePath + localBranch + "/Cache.txt"  # Where you saved the Cache txt file
CSVsummaryFileName = CSVPath + "summary2.csv"
analogyFileName = CSVPath + "something.csv"
list_of_mapped_preds = []

print("\nINPUT:", localPath)
print("OUTPUT:", CSVPath)

sourceFiles = os.listdir(localPath)
# all_csv_files = [i for i in sourceFiles if i.endswith('txt.S.csv')] # if ("code" in i) and (i.endswith('.csv'))]
all_csv_files = [i for i in sourceFiles if i.endswith(filetypeFilter)]
all_csv_files.sort(reverse=True)
# all_csv_files = [i for i in sourceFiles if i.endswith(filetypeFilter) and i.startswith('S')]
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
    prep_list = ['of', 'for', 'at', 'on', 'as', 'by', 'in', 'to', 'from', 'into', 'through', 'toward', 'with']
    return word in prep_list


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


def graph_contains_proper_noun_UNUSED(propNoun1):
    global temp_graph2
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


def build_graph_from_csv_DEPRECATED(file_name):
    """ Includes eager concept fusion rules. Enforces  noun_properNoun_pronoun"""
    global temp_graph2  # an ORDERED multi Di Graph, converted at the end
    global term_separator
    global invalid_graphs
    fullPath = localPath + file_name
    unknownCounter = 1
    print("BGC")
    with open(fullPath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp_graph2.clear()
        temp_graph2.graph['Graphid'] = file_name
        try:
            previous_subj = last_v = previous_obj = ""
            for row in csvreader:
                if len(row) == 3:  # subject, verb, obj
                    noun1, verb, noun2 = row
                    noun1 = noun1.strip()
                    verb = verb.strip()
                    noun2 = noun2.strip()
                    if skip_prepositions and prepositionTest(verb):
                        continue
                    if noun1.lower() == 'unknown' or noun1.lower()[0:9] == '[unknown]':
                        noun1 = chr(ord('@') + unknownCounter) + "_" + "unknown"  # Unique proper nouns
                        unknownCounter += 1
                        continue
                    if noun2.lower() == 'unknown' or noun2.lower()[0:9] == '[unknown]':
                        noun2 = chr(ord('@') + unknownCounter) + "_" + "unknown"
                        unknownCounter += 1
                        continue
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
                    # elif row[0] == "CodeContracts":  # skip the contracts?
                    #    break
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
        eager_concept_fusion()
    returnGraph = nx.MultiDiGraph(temp_graph2)
    return returnGraph  # results in the canonical  version of the graph :-)
# end of build_graph_from_csv(targetFile)


def build_graph_from_csv(file_name):
    """ Includes eager concept fusion rules. Enforces  noun_properNoun_pronoun"""
    global temp_graph2  # an ORDERED multi Di Graph, converted at the end
    global term_separator
    global invalid_graphs
    fullPath = localPath + file_name
    #print("BGC2")
    with open(fullPath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp_graph2.clear()
        temp_graph2.graph['Graphid'] = file_name
        try:
            previous_subj = last_v = previous_obj = ""
            for triple in csvreader:
                if len(triple) == 3:  # subject, verb, obj
                    noun1, verb, noun2 = triple
                elif len(triple) == 4:  # methodName, subject, verb, obj
                    methodName, noun1, verb, noun2 = triple
                elif len(triple) == 0:
                    continue
                elif mode == 'code' or len(triple) == 6:  # Code Graphs
                    if triple[0] == "CodeContracts":  # skip the contracts?
                        pass  # break
                    if len(triple) == 6:
                        methodName, noun1, verb, noun2, nr1, nr2 = triple
                        noun1 = noun1.strip() + term_separator + nr1
                        noun2 = noun2.strip() + term_separator + nr2
                    elif len(triple) == 3:
                        exit("BGfC len  triple ==  3 error")
                        noun1, verb, noun2 = triple
                    elif len(triple) != 3:
                        print("Possibly embedded SQL in: ")
                        pass
                    else:
                        noun1, verb, noun2 = triple
                noun1 = noun1.strip()     # remove spaces
                verb = verb.strip()
                noun2 = noun2.strip()
                if mode == 'English':
                    if skip_prepositions and prepositionTest(verb):
                        continue
                    if (noun1 == previous_subj) and (noun2 == previous_obj) and prepositionTest(verb):
                        verb = last_v + "_" + verb
                        print(verb, end=" ")
                        temp_graph2.remove_edge(noun1, noun2)
                    elif noun1 == "NOUN":  # skip header information
                        continue

                    if len(noun1.split(term_separator)) > 1:
                        noun1 = parse_new_coref_chain(noun1)
                    if len(noun2.split(term_separator)) > 1:
                        noun2 = parse_new_coref_chain(noun2)

                temp_graph2.add_node(noun1, label=noun1)
                temp_graph2.add_node(noun2, label=noun2)
                temp_graph2.add_edge(noun1, noun2, label=verb)

                if mode == 'English':
                    previous_subj = noun1
                    last_v = verb  # phrasal verbs; read_over, read_up, read_out 
                    previous_obj = noun2
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (CSVPath, csvreader.line_num, e))
    if mode == 'English':
        print(temp_graph2.number_of_nodes(), end=" -> ")
        eager_concept_fusion2(temp_graph2)
        print(temp_graph2.number_of_nodes(), end=" ")
    returnGraph = nx.MultiDiGraph(temp_graph2) # no need for a .copy()
    return returnGraph  # results in the canonical  version of the graph :-)


def build_graph_from_triple_list(triple_list):
    previous_subj = last_v = previous_obj = ""
    temp_graph2 = nx.MultiDiGraph()
    temp_graph2.clear()
    temp_graph2.graph['Graphid'] = "Dud Name"
    for triple in triple_list:
        if len(triple) == 3:  # subject, verb, obj
            noun1, verb, noun2 = triple
        elif len(triple) == 4:  # methodName, subject, verb, obj
            methodName, noun1, verb, noun2 = triple
        elif len(triple) == 0:
            continue
        elif mode == 'code' or len(triple) == 6:  # Code Graphs
            if triple[0] == "CodeContracts":  # skip the contracts?
                pass  # break
            if len(triple) == 6:
                methodName, noun1, verb, noun2, nr1, nr2 = triple
                noun1 = noun1.strip() + term_separator + nr1
                noun2 = noun2.strip() + term_separator + nr2
            elif len(triple) == 3:
                exit("BGfC len  triple ==  3 error")
                noun1, verb, noun2 = triple
            elif len(triple) != 3:
                print("Possibly embedded SQL in: ")
                pass
            else:
                noun1, verb, noun2 = triple
        noun1 = noun1.strip()  # remove spaces
        verb = verb.strip()
        noun2 = noun2.strip()
        if mode == 'English':
            if skip_prepositions and prepositionTest(verb):
                continue
            if (noun1 == previous_subj) and (noun2 == previous_obj) and prepositionTest(verb):
                verb = last_v + "_" + verb
                print(verb, end=" ")
                temp_graph2.remove_edge(noun1, noun2)
            elif noun1 == "NOUN":  # skip header information
                continue

            if len(noun1.split(term_separator)) > 1:
                noun1 = parse_new_coref_chain(noun1)
            if len(noun2.split(term_separator)) > 1:
                noun2 = parse_new_coref_chain(noun2)

        temp_graph2.add_node(noun1, label=noun1)
        temp_graph2.add_node(noun2, label=noun2)
        temp_graph2.add_edge(noun1, noun2, label=verb)

        if mode == 'English':
            previous_subj = noun1
            last_v = verb  # phrasal verbs; read_over, read_up, read_out
            previous_obj = noun2
    if mode == 'English':
        print(temp_graph2.number_of_nodes(), end=" -> ")
        eager_concept_fusion2(temp_graph2)
        print(temp_graph2.number_of_nodes(), end=" ")
    returnGraph = nx.MultiDiGraph(temp_graph2) # no need for a .copy()
    return returnGraph  # results in the canonical  version of the graph :-)


def parse_new_coref_chain(in_chain):
    """ For silly long coref chains. """
    if mode == 'Code':
        return in_chain
    cnt = in_chain.find(term_separator)
    if cnt < 0:
        return in_chain
    else: # cnt <= 100:  # parser works poorly on short noun sequences
        return reorganise_coref_chain(in_chain)
    # elif mode == "English":  # parse_coref_subsentence()
    #     noun_lis = []
    #     propN_lis = []
    #     pron_lis = []
    #     chn = in_chain.split(term_separator)  # "its_warlike_neighbor_Gagrach".split("_")
    #     strg2 = " ".join(chn)
    #     chn2 = nltk.pos_tag(word_tokenize(strg2))
    #     for (tokn, po) in chn2:
    #         if tokn in ['a', 'the', 'its']:  # remove problematic words from coref phrases
    #             continue
    #         elif is_pronoun(tokn) or po == "PRP":
    #             pron_lis.append(tokn)
    #         elif po == "NN" or is_noun(tokn):
    #             noun_lis.append(tokn)
    #         elif po == "NNP" or is_proper_noun(tokn):
    #             propN_lis.append(tokn)
    #     res = noun_lis + propN_lis + pron_lis
    #     slt = "_".join(res)
    #     return slt
    return "ERROR - parse_new_coref_chain() "


def reorganise_coref_chain(strg):  # noun-propernoun-pronoun   #w, tag = nltk.pos_tag([wrd])[0]
    global term_separator
    noun_lis, propN_lis, pron_lis, possible_propN_lis = [], [], [], []
    noun_lis2, propN_lis2, pron_lis2, possible_propN_lis2 = [], [], [], []
    if strg.find(term_separator) < 0:
        slt = strg
    else:
        chan = strg.replace(term_separator, " ")
        text = nltk.word_tokenize(chan)
        pos_tag_list = nltk.pos_tag(text)
        for w, tokn in pos_tag_list:
            if w in ['a', 'the', 'its', 'A', 'The', 'Its']:  # remove problematic words
                continue
            elif tokn == 'PRON':
                pron_lis.append(w)
                pron_lis2.append([w, tokn])
            elif tokn in ["N", "NN", "NNS"]:
                noun_lis.append(w)
                noun_lis2.append([w, tokn])
            elif tokn in ['NP']:
                propN_lis.append(w)
                propN_lis2.append([w, tokn])
            else:
                possible_propN_lis.append(w)
                possible_propN_lis2.append([w, tokn])
        # print(noun_lis2 + propN_lis2 + possible_propN_lis2 + pron_lis2, end="   ")
        slt = "_".join(noun_lis + propN_lis + possible_propN_lis + pron_lis)
    return slt
# reorganise_coref_chain("its_warlike_neighbor_Gagrach")
# reorganise_coref_chain("He_hunter_he_him")
# reorganise_coref_chain('hawk_she_Karla')
# reorganise_coref_chain('hawk_Karla_she')


def eager_concept_fusion():
    global term_separator
    global temp_graph2
    global coalescing_completed
    if mode == 'code':
        return
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


def delete_all_duplicates(seq):
    seen = {}
    pos = 0
    for item in seq:
        if item not in seen:
            seen[item] = True
            seq[pos] = item
            pos += 1
    del seq[pos:]


def remove_duplicates(lis):
    return list(set(lis))


def flatten_list(t):   # single level of nesting removed
    return [item for sublist in t for item in sublist]


def cascade_concept_merge(lis):  #cascade_concept_merge(['twin_himself_he', 'twin_One_himself_he'])
    if lis == nil or type(list)==None:
        return
    for graph_node in lis[0].split(term_separator):  # chains already reorganised
        for graph_node2 in node_list[1 + node_list.index(graph_node)]:
         if graph_node.split("_")[0] == graph_node2.split("_")[0]:
             exit(" Problems!! ")
    return null


def eager_concept_merge_rules(node_list):
    coalescing_completed = True
    merge_condition_detected = False
    list_of_removed_nodes = []
    nodes_to_be_merged = []
    for graph_node in node_list:  # node_list:  # Final-pass Coalescing
        # gn_n = contains_noun(graph_node)
        if graph_node == 'twin':
            xxxx = 0
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        if "_" in graph_node:
            new_node_name = reorganise_coref_chain(graph_node)
            if new_node_name != graph_node:
                graph_node = new_node_name
                # graph_node_reogrganised = True
        gn_word_list = graph_node.split(term_separator)
        extendedNoun = ""
        for graph_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes only
            if graph_node is graph_node2 or graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            intersect = intersection(gn_word_list, gn2_word_list)
            if len(intersect) > 0 and contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and intersection(gn_pn, gn2_pn):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:  # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes \
                    and not graph_node2 in list_of_removed_nodes:
                if extendedNoun == "":
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                else:
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    nodes_to_be_merged.append([graph_node2, extendedNoun])
                    list_of_removed_nodes.append(graph_node2)
                if not (graph_node == extendedNoun):
                    nodes_to_be_merged.append([graph_node, extendedNoun])
                    list_of_removed_nodes.append(graph_node)
                merge_condition_detected = False
                # coalescing_completed = False
    return nodes_to_be_merged   # ordered


def return_eager_merge_candidates(inGraph):
    global term_separator
    global coalescing_completed
    global temp_graph2
    node_list = list(inGraph.nodes())
    #if node_list != [] and node_list[0] == 'twin':
    #    xxx = 0
    coalescing_completed = True    ################################################################################
    merge_condition_detected = False
    list_of_removed_nodes = []
    nodes_to_be_merged = []
    reorganise_condition_detected = False
    for graph_node in node_list:  # eager_concept_merge_rules(node_list)
        # gn_n = contains_noun(graph_node)
        if graph_node == 'twin' or graph_node == 'tail_feathers_sportsman_her_he':
            xxxx = 0
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        if "_" in graph_node:
            new_node_name = reorganise_coref_chain(graph_node)
            if new_node_name != graph_node:
                # graph_node = new_node_name
                reorganise_condition_detected = True
        gn_word_list = graph_node.split(term_separator)
        extendedNoun = ""
        for graph_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes only
            if graph_node is graph_node2 or graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            intersect = intersection(gn_word_list, gn2_word_list)
            if len(intersect) > 0 and contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and intersection(gn_pn, gn2_pn):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:  # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes \
                    and not graph_node2 in list_of_removed_nodes:
                if extendedNoun == "":
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                else:
                    extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    nodes_to_be_merged.append([graph_node2, extendedNoun])
                    list_of_removed_nodes.append(graph_node2)
                if not (graph_node == extendedNoun):
                    nodes_to_be_merged.append([graph_node, extendedNoun])
                    list_of_removed_nodes.append(graph_node)
            if reorganise_condition_detected and not merge_condition_detected:
                nodes_to_be_merged.append([graph_node, new_node_name])
                list_of_removed_nodes.append(graph_node)
                reorganise_condition_detected = False
            merge_condition_detected = False
            # coalescing_completed = False     ################################################################################
    return nodes_to_be_merged


def eager_concept_fusion2(in_graph):  # EAGER CONCEPT FUSION of parsed nodes
    global temp_graph2
    nodes_to_be_merged = "dummy-value"
    iter_count = 0
    original_number_of_nodes = in_graph.number_of_nodes()
    while nodes_to_be_merged != [] and iter_count<1:  # single repretition of cascading
        if in_graph.graph['Graphid'][0:22] == 'TumFort - The Identical Twins.txt_v3.txt.dcorf..csv'[0:22]:
            xxxx = 0
        nodes_to_be_merged = return_eager_merge_candidates(in_graph)
        for graph_node, extendedNoun in nodes_to_be_merged:     # execute concept merge
            if not in_graph.has_node(graph_node):
                continue
            print(" MERGE ", graph_node, "->", extendedNoun, end="--    ")
            remapping = {graph_node: extendedNoun}
            in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
            in_graph.nodes[extendedNoun]['label'] = extendedNoun
        iter_count += 1
    #outside main while loop
    num_removed_nodes = original_number_of_nodes - in_graph.number_of_nodes()
    if original_number_of_nodes - in_graph.number_of_nodes() > 0:
        print(" ECF2 reduction by ", num_removed_nodes, end=" nodes.  ")
    temp_graph2 = in_graph.copy()
    return


def eager_concept_fusion2_BACKUP2(in_graph):  # EAGER CONCEPT FUSION of parsed nodes
    global term_separator
    global temp_graph2
    node_list = list(in_graph.nodes())
    merge_condition_detected = False
    original_number_of_nodes = in_graph.number_of_nodes()
    list_of_removed_nodes = []
    indx1 = 0
    while indx1 < len(node_list):  # node_list:  # Final-pass Coalescing
        graph_node = node_list[indx1]    # gn_n = contains_noun(graph_node)
        if graph_node == 'twin':
            xxxx=0
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        gn_word_list = graph_node.split(term_separator)
        indx2 = indx1 + 1
        while indx2 < len(node_list):
        #for graph_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes only
            graph_node2 = node_list[indx2]
            if graph_node is graph_node2 or graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            intersect = intersection(gn_word_list, gn2_word_list)
            if intersect and contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and intersection(gn_pn, gn2_pn):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:       # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes\
                and not graph_node2 in list_of_removed_nodes:
                extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    print(" FPC2 ", graph_node,"+",graph_node2, "->", extendedNoun, end="--    ")
                    remapping = {graph_node2: extendedNoun}
                    in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
                    in_graph.nodes[extendedNoun]['label'] = extendedNoun
                    list_of_removed_nodes.append(graph_node2)
                if not graph_node == extendedNoun:
                    print(" FPC2 ", graph_node,"+",graph_node2, "->", extendedNoun, end="--    ")
                    remapping = {graph_node: extendedNoun}  # merge in graph_node too
                    in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
                    in_graph.nodes[extendedNoun]['label'] = extendedNoun
                    list_of_removed_nodes.append(graph_node)
                    graph_node = extendedNoun # subsequence node-merges will be correct & faster
                    gn_word_list = graph_node.split(term_separator)
                merge_condition_detected = False
            indx2 += 1
        indx1 += 1
    r = original_number_of_nodes - in_graph.number_of_nodes()
    print("FPC2 reduction by ", original_number_of_nodes - in_graph.number_of_nodes(), end=" nodes.  ")
    temp_graph2 = in_graph.copy()
    return


def eager_concept_fusion2_ZZZ(in_graph):  # final_pass_coalescing()  final_pass_coalescing2() EAGER CONCEPT FUSION of parsed nodes
    global term_separator
    global temp_graph2
    node_list = list(in_graph.nodes())
    merge_condition_detected = False
    original_number_of_nodes = in_graph.number_of_nodes()
    list_of_removed_nodes = []
    indx1 = 0
    nodes_to_be_merged = []
    while indx1 < len(node_list):  # node_list:  # Final-pass Coalescing
        graph_node = node_list[indx1]    # gn_n = contains_noun(graph_node)
        if graph_node == 'twin':
            xxxx=0
        gn_pn = contains_proper_noun2(graph_node)
        if graph_node in list_of_removed_nodes:
            continue
        gn_word_list = graph_node.split(term_separator)
        indx2 = indx1 + 1
        while indx2 < len(node_list):
        #for graph_node2 in node_list[1 + node_list.index(graph_node):]:  # subsequent nodes only
            graph_node2 = node_list[indx2]
            if graph_node is graph_node2 or graph_node2 in list_of_removed_nodes:
                continue
            gn2_pn = contains_proper_noun2(graph_node2)
            gn2_word_list = graph_node2.split(term_separator)
            intersect = intersection(gn_word_list, gn2_word_list)
            if intersect and contains_noun("_".join(intersect)):
                merge_condition_detected = True
            if not merge_condition_detected and gn_pn and gn2_pn and intersection(gn_pn, gn2_pn):
                merge_condition_detected = True
            elif not merge_condition_detected and head_word(graph_node) == head_word(graph_node2):
                r = intersecting_proper_nouns(gn_pn, gn2_pn)
                if r:       # and not different proper nouns, or incompatible pronouns
                    merge_condition_detected = True
            if merge_condition_detected and not (graph_node == graph_node2) and not graph_node in list_of_removed_nodes\
                and not graph_node2 in list_of_removed_nodes:
                extendedNoun = extend_as_set(graph_node.split(term_separator), graph_node2.split(term_separator))
                extendedNoun = reorganise_coref_chain(extendedNoun)
                if not graph_node2 == extendedNoun:
                    #print(" FPC2 ", graph_node,"+",graph_node2, "->", extendedNoun, end="--    ")
                    #remapping = {graph_node2: extendedNoun}
                    #in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
                    #in_graph.nodes[extendedNoun]['label'] = extendedNoun
                    list_of_removed_nodes.append(graph_node2)
                if not graph_node == extendedNoun:
                    #print(" FPC2 ", graph_node,"+",graph_node2, "->", extendedNoun, end="--    ")
                    #remapping = {graph_node: extendedNoun}  # merge in graph_node too
                    #in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
                    #in_graph.nodes[extendedNoun]['label'] = extendedNoun
                    list_of_removed_nodes.append(graph_node)
                    #graph_node = extendedNoun # subsequence node-merges will be correct & faster
                    #gn_word_list = graph_node.split(term_separator)
                merge_condition_detected = False
            indx2 += 1
        indx1 += 1
        for graph_node, extendedNoun in nodes_to_be_merged:
            print(" MERGE ", graph_node, "+", graph_node2, "->", extendedNoun, end="--    ")
            remapping = {graph_node: extendedNoun}
            in_graph = nx.relabel_nodes(in_graph, remapping, copy=False)
            in_graph.nodes[extendedNoun]['label'] = extendedNoun
    r = original_number_of_nodes - in_graph.number_of_nodes()
    print("FPC2 reduction by ", original_number_of_nodes - in_graph.number_of_nodes(), end=" nodes.  ")
    temp_graph2 = in_graph.copy()
    return


def intersecting_proper_nouns(list1, list2):
    if list1 and list2:
        return intersection(list1, lis2)
    elif not list1 == False and not list2 == False:
        return True
    else:
        return False


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def list_diff(li1, li2):
    return (list(set(li1) - set(li2)))


# ####################################################################################
# ####################################################################################
# ####################################################################################


def head_word(term):  # first word before term separator
    global term_separator
    z = term.split(term_separator)[0]
    return z


def contains_noun(chain):  # noun... PropN ... Pron
    if not(chain):
        return False
    global term_separator
    propN_lis = []
    wrd_lis = chain.split(term_separator)
    pos_tag_list = nltk.pos_tag(wrd_lis)
    if chain == 'bulk':
        print(chain, end="--- ")
    indx = 0
    while indx < len(wrd_lis) and (wn.synsets(wrd_lis[indx], pos=wn.NOUN)
                                   or pos_tag_list[indx][1] in ['N', 'NN']): # common nouns
        propN_lis.append(wrd_lis[indx])
        indx +=1
    if propN_lis == []:
        return []
    else:
        return propN_lis


def is_noun(wrd):
    zz = wn.synsets(wrd, pos=wn.NOUN)
    return zz != []


def is_proper_noun_DEPRECATED(wrd):
    zz = nltk.pos_tag([wrd])
    return ((nltkwords.words().__contains__(wrd) == False)
            and (wn.synsets(wrd) == []))
# is_proper_noun_DEPRECATED("Bezos")   is_proper_noun_DEPRECATED("karla")

def is_proper_noun(wrd):
    w, tag = nltk.pos_tag([wrd])[0]
    return tag == "NNP" or tag == "NP"
# is_proper_noun("Karla")   is_proper_noun("karla")

def contains_proper_noun_DEPRECATED(chain):  # noun... PropN ... Pron
    global term_separator
    for wo in chain.split(term_separator):
        if ((nltkwords.words().__contains__(wo) == False) and (wn.synsets(wo) == [])):
            return wo
    return False


def contains_proper_noun(chain):  # noun... PropN ... Pron
    global term_separator
    propN_lis = []
    wrd_lis = chain.split(term_separator)
    pos_tag_list = nltk.pos_tag(wrd_lis)
    indx = 0
    while indx < len(wrd_lis) and (wn.synsets(wrd_lis[indx], pos=wn.NOUN)
                                   or pos_tag_list[indx][1] in ['N', 'NN']): # common nouns
        indx +=1
    while indx < len(wrd_lis) and pos_tag_list[indx][1] in ['NNP', 'NP']:  # Proper nouns in the middle
        propN_lis.append(pos_tag_list[indx])
        print("PropN:", pos_tag_list[indx][0], end=" ")
        indx += 1
    if propN_lis == []:
        return []
    else:
        return propN_lis[0][0]
# contains_proper_noun("Karla")   contains_proper_noun("she_karla")
# contains_proper_noun("We_John") contains_proper_noun("It_Boeing")
# contains_proper_noun("We_John_Johns")

def contains_proper_noun2(strg):  # parsing reorgansied chains Sucks!
    chan = strg.replace(term_separator, " ")
    text = nltk.word_tokenize(chan)
    pos_tag_list = nltk.pos_tag(text)
    propN_list = []
    for w,tag in pos_tag_list:
        if tag in ['NP', 'NNP']:
            if not w in propN_list:
                propN_list.append(w)
    if propN_list == []:
        return False
    else:
        return propN_list
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
    s_tot = v_tot = o_tot = 0
    s_count = v_count = o_count = 0
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

def SOMETHING_ELSE_ratio_of_mapped_source_predicates(tgt):  # returnMappingRatio(sourceGraph)
    number_mapped_preds = number_unmapped_preds = 0
    s_tot = v_tot = o_tot = 0
    s_count = v_count = o_count = 0
    lis = returnEdgesAsList(tgt)
    for (s, v, o) in lis:
        if (s in GM.mapping.values()) and (v in GM.mapping.values()) and (o in GM.mapping.values()):
            number_mapped_preds += 1
            s_sim = wn_sim_mine(s, GM.mapping[s], 'n')[0]
            v_sim = wn_sim_mine(v, GM.mapping[v], 'v')[0]
            o_sim = wn_sim_mine(o, GM.mapping[o], 'n')[0]
            if s_sim > 0:
                s_tot += s_sim
                s_count += 1
            if v_sim > 0:
                v_tot += v_sim
                v_count += 1
            if o_sim > 0:
                o_tot += o_sim
                o_count += 1
        else:
            number_unmapped_preds += 1
    if s_count > 0:
        s_avg = s_tot / s_count
    if v_count > 0:
        v_avg = v_tot / v_count
    if o_count > 0:
        o_avg = o_tot / o_count
    return number_mapped_preds, number_unmapped_preds, s_avg, v_avg, o_avg


def printMappedPredicates_UNORDERED(graf):  # printMappedPredicates(sourceGraph)
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
                rslt = wn_sim_mine(v, GM.mapping.get(v), 'v')
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


def printMappedPredicates(graf):  # printMappedPredicates(targetGraph)
    # print mapped predicates first, then unmapped ones""
    mapped = notMapped = 0
    lis = returnEdgesAsList(graf)
    global list_of_mapped_preds
    global analogyFilewriter
    t_list = s_list = []
    flag = False
    unmapped = set()
    out_list = []
    for v in list_of_mapped_preds:
        s, t, u = v
        t_list.append(t)
        s_list.append(s)
        t_s, t_v, t_o = t
        s_s, s_v, s_o = s                # Full predicate mapped
        print("{: <20.20} {: >1} {: <10.10} {: >1} {: >20.20}".format(t_s, " ", t_v, " ", t_o), end="  ==   ")
        print("{: <20.20} {: >1} {: <10.10} {: >1} {: >20.20}".format(s_s, " ", s_v, " ", s_o), end="   ")
        rslt = wn_sim_mine(t_v, s_v, 'v')
        tmp = round((float(rslt[0]) + (float(rslt[2]))) / 2, 2)
        print('{:>3.1f}'.format(tmp), " ", simplifyLCS(rslt[1]), simplifyLCS(rslt[3]))
        out_list = [t_s, t_v, t_o, "    ==    ", s_s, s_v, s_o, tmp]
        analogyFilewriter.writerow(out_list)
    unmapped_target_preds = 0
    for s, v, o in lis:
        if [s, v, o] in s_list:
            continue
        else:
            unmapped_target_preds +=1
        s_mapped = s in GM.mapping.keys()
        v_mapped = v in GM.mapping.keys()
        o_mapped = o in GM.mapping.keys()
        out_list = [s, v, o, "    ==    "]
        if not s_mapped and not v_mapped and not o_mapped:
            # print("{: <20} {: >1} {: >10} {: >1} {: >20}".format(" _ ", " ", " _ ", " ", " _ "))
            continue
        print("{: <20.20} {: >1} {: >10.10} {: >1} {: >20.20}".format(s, " ", v, " ", o), end="  ==   ")  # Target pred
        if s_mapped:  # partial predicate mapping
            print("{: <20.20}".format(GM.mapping.get(s)), end=" ")
            out_list.append(GM.mapping.get(s))
        else:
            print("{: >20.20}".format("_"), end="")
            out_list.append(" _")
            unmapped.add(s)
        if v_mapped:
            print("{: >10.10}".format(GM.mapping.get(v)), end=" ")
            out_list.append(GM.mapping.get(v))
        else:
            print("{: >10.10}".format("_ "), end="")
            out_list.append(" _ ")
            unmapped.add(v)
        if o_mapped:
            print("{: >20.20}".format(GM.mapping.get(o)))
            out_list.append(GM.mapping.get(o))
        else:
            print("{: >20.20}".format("_"))
            out_list.append(" _ ")
            unmapped.add(o)
        if out_list:
            analogyFilewriter.writerow(out_list)
    print("UNMAPPED Target preds", unmapped_target_preds, " of ",  graf.number_of_edges(), "    UNMAPPED: ", unmapped)
    analogyFilewriter.writerow(["UNMAPPED: ", str(unmapped).replace(",", " ")])
    return mapped, notMapped  # numeric summary


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


# Shows graph in html/javascript D3js force directed graph.
def show_graph_in_FF(Graph):
    data = {
        'nodes': [],
        'edges': []
    }
    for index, node in enumerate(Graph.nodes()):
        print(index, node, end="  ")
        Graph.nodes[node]['index'] = index  # some nodes may not have label attribute set
    print("", end=" j j j ")
    for node_id, attr in Graph.nodes(data=True):
        print(node_id, attr, end="  ")
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
    #webbrowser.open(os.path.realpath(htmlBranch + graphName + '.html'), new=2)

    webbrowser.open_new_tab( os.path.realpath(htmlBranch + graphName + '.html') )


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
    if os.path.isfile(CachePath):
        with open(CachePath, "r") as wn_cache_file:
            filereader = csv.reader(wn_cache_file)
            for row in filereader:
                try:
                    WN_cache[row[0] + "-" + row[1]] = row[2:]
                except IndexError:
                    pass
read_wn_cache_to_dict()
print("WordNet Cache initialised.   ", end="")


def wn_sim_mine(w1, w2, partoS):
    # wn_sim_mine("create","construct", 'v') -> [0.6139..., 'make(v.03)', 0.666..., 'make(v.03)'] """
    global LCSLlist
    lin_max = wup_max = 0
    LCSL_temp = LCSW_temp = []
    w1 = w1.lower()
    w2 = w2.lower()
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    if wn1 is None:
        wn1 = w1
    if wn2 is None:
        wn2 = w2
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
                    LCSW_temp = ss1.lowest_common_hypernyms(ss2)  # may return []
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
    if not LCSL_temp:
        LCSL = "Synset('null." + partoS + ".0303')"
    if not LCSW_temp:
        LCSW = "Synset('null." + partoS + ".0404')"
    return [lin_max, LCSL, wup_max, LCSW]


def wn_sim_mine(w1, w2, partoS, use_lexname=True):
    """ wn_sim_mine("create","construct", 'v') -> [0.6139..., 'make(v.03)', 0.666..., 'make(v.03)'] """
    global LCSLlist
    lexname_out = ""
    lin_max = wup_max = 0
    LCSL_temp = LCSW_temp = []
    flag = False
    w1 = w1.lower()
    w2 = w2.lower()
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    if wn1 is not None:
        w1 = wn1
    if wn2 is not None:
        w2 = wn2
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
            if use_lexname:
                ss1_lex_nm = ss1.lexname()
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
                    LCSW_temp = ss1.lowest_common_hypernyms(ss2)  # may return []
                if lin is None:
                    lin = 0
                if wup is None:
                    wup = 0
                if use_lexname and flag ==  False:
                    if ss1_lex_nm == ss2.lexname() and flag == False:
                        lexname_out = ss1_lex_nm
                        flag = True
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
        write_to_wn_cache_file(w1, w2, partoS, lin_max, wup_max, LCSL, LCSW+"-"+lexname_out)
    LCSLlist.append(LCSL)  # for the GUI presentation
    if LCSL_temp == []:
        LCSL = "Synset('null." + partoS + ".0404')"
    if LCSW_temp == []:
        LCSW = "Synset('null." + partoS + ".0403')"
    return [lin_max, LCSL, wup_max, LCSW+" "+lexname_out]
    # return [lin_max, LCSL, wup_max, LCSW, lexname_out]
# wn_sim("create","construct", 'v')


def write_to_wn_cache_file(w1, w2, pos, Lin, Wup, L_lcs, W_lcs):
    global WN_cache
    global CachePath
    if w1 == w2:
        return
    # if bool(WN_cache) and WN_cache.has_key(w1+'-'+w2):
    if bool(WN_cache) and w1+'-'+w2 in WN_cache:
            return
    else:
        with open(CachePath, "a+") as wn_cache_file:
            Stringtest = w1 + "," + w2 + "," + pos
            Stringtest += "," + str(Lin) + "," + str(Wup) + "," + L_lcs + "," + W_lcs   # + ","
            wn_cache_file.write(" \n" + Stringtest)


###########################################################################################################

def calculate2Similarities(tgt_graph, source_graph, best_analogy):  # source concept nodes
    global arrow
    global analogyFilewriter
    global list_of_mapped_preds
    global semcor_ic
    global result_graph
    d = 0
    j = 0
    i = 0
    max_value = 0.0
    max_value2 = 0.0
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

    # global number_of_inferences
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
              " {:.2f}".format(avg_Lin_conceptual), " {:.2f}".format(avg_Wup_conceptual), conSim[2], " ",
              " {:.2f}".format(avg_Lin_relational), " {:.2f}".format(avg_Wup_relational),
              relSim[2], " ", relSim[3], " ", number_of_inferences, " ", mappedConcepts, " ", mappedRelations, " ",
              " {:.2f}".format(average_relational_similarity),
              GM.mapping['Number_Mapped_Predicates'], GM.mapping['Total_Score'] )

        list_of_mapped_preds

        list_of_subgraphs = list(nx.weakly_connected_components(result_graph))
        list_of_digraphs = []
        for subgraph in list_of_subgraphs:
            list_of_digraphs.append(nx.subgraph(result_graph, subgraph))
        if nx.weakly_connected_components(result_graph) == 0:
            print(" nx.weakly_connected_components(result_graph) ")
        if result_graph.number_of_nodes() > 0:
            max_wcc = max(nx.weakly_connected_components(result_graph), key=len)
        else:
            max_wcc = []
        # ("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "%Map",
        # "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "#Infs", "AvgRelSim", "LargCpnnt", "#ConnCmpnnt", "Score")
        writeSummaryFileData(targetGraph.graph['Graphid'], sourceGraph.graph['Graphid'],
                             tgt_graph.number_of_nodes(), tgt_graph.number_of_edges(),
                             source_graph.number_of_nodes(), source_graph.number_of_edges(),
                             len(best_analogy),  (len(GM.mapping)- len(best_analogy)),
                             avg_Lin_conceptual, avg_Wup_conceptual, avg_Lin_relational, avg_Wup_relational,
                             number_of_inferences, mappedConcepts, average_relational_similarity,
                             len(max_wcc), len(list_of_digraphs), GM.mapping['Total_Score'] )
        printMappedPredicates(tgt_graph)
    # analogyFile.flush()
    # analogyFile.close()


def calculate2Similarities_UNUSED(tgt_graph, source_graph, best_analogy):
    global analogyFilewriter
    global list_of_mapped_preds
    global list_of_inferences
    global GM
    global CSVPath
    global analogyFileName
    global CachePath
    conceptual_sim = numpy.zeros(7)
    relational_sim = numpy.zeros(7)
    predicate_count = 0
    for t,s in list_of_mapped_preds:
        t_sub, tRel, t_obj = t
        s_sub, sRel, s_obj = s
        predicate_count +=1
        r1 = evaluateRelationalSimilarity(tRel, sRel)    # numpy.zeros(7) Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
        r2 = evaluateConceptualSimilarity(t_sub, s_sub)
        r3 = evaluateConceptualSimilarity(t_obj, s_obj)
        conceptual_sim = np.add(conceptual_sim, r1)
        relational_sim = np.add(relational_sim, r2)
        conceptual_sim = np.add(conceptual_sim, r3)

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
            reslt[5] = 0.81
            LinLCS = tRel.split(term_separator)[:1]
            WuPLCS = tRel.split(term_separator)[:1]
        elif tRel.split(term_separator)[0] == sRel.split(term_separator)[0]:
            reslt[4] = 0.33
            reslt[5] = 0.33
            LinLCS = tRel.split(term_separator)[0]
            WuPLCS = tRel.split(term_separator)[0]
        else:
            temp_result = wn_sim_mine(tRel, sRel, 'v')
            # print("temp_result",temp_result)
            reslt[4] = temp_result[0]
            reslt[5] = temp_result[2]
            LinLCS = 'no-reln'
            WuPLCS = 'no-reln'
    else:
        temp_result = wn_sim_mine(tRel, sRel, 'v')  # returns [0.6139, 'make(v.03)', 0.666, 'make(v.03)']
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


def evaluateConceptualSimilarity(tConc, sConc):  # ('cat','dog')
    global term_separator
    reslt = numpy.zeros(7)  # Lin0, WuP0, Lin1, Wup1, LinSum, WuPSum,
    if tConc == sConc:
        reslt[2] = 1
        reslt[3] = 1
        reslt[4] = 1
        reslt[5] = 1
        LinLCS = tConc
        WuPLCS = tConc
    elif mode == 'code':
        if second_head(tConc) == second_head(sConc):
            reslt[4] = 0.95
            reslt[5] = 0.95
            LinLCS = second_head(tConc)
            WuPLCS = LinLCS
        elif tConc.split(term_separator)[:1] == sConc.split(term_separator)[:1]:
            reslt[4] = 0.8
            reslt[5] = 0.8
            LinLCS = tConc.split(term_separator)[:1]
            WuPLCS = tConc.split(term_separator)[:1]
        elif tConc.split(term_separator)[0] == sConc.split(term_separator)[0]:
            reslt[4] = 0.33
            reslt[5] = 0.33
            LinLCS = tConc.split(term_separator)[0]
            WuPLCS = tConc.split(term_separator)[0]
        else:
            reslt[0] = 0.00001
            reslt[1] = 0.00001
            LinLCS = 'none'
            WuPLCS = 'none'
    else:
        temp_result = wn_sim_mine(tConc, sConc, 'n')  # returns [0.6139 'make(v.03)', 0.666, 'make(v.03)']
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
    elif str(synsetName)[:6] == "Synset":  # instance of <class 'nltk.corpus.reader.wordnet.Synset'>
        ssString = str(synsetName)
        y = ssString.find('(') + 2
        z = ssString.find('.') + 5
        synsetName = ssString[y:z].replace('.', '(', 1) + ")"
    return synsetName  # [synsetName]


# simplifyLCSList("[Synset('whole.n.02')]")

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


# #######################################################################
# ############################   Graph   ################################
# ############################  Matching   ##############################
# #######################################################################

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
    #GM.mapping.clear()
    isomorphvf2CB.temp_sol = []
    before_seconds = time.time()
    total_sim_score = 0
    if algorithm == "DFS":
        #list_of_mapped_preds, relato_struct_sim = DFS.generate_and_explore_mapping_space(sourceGraph, targetGraph)
        list_of_mapped_preds, number_mapped_predicates = DFS.generate_and_explore_mapping_space(targetGraph, sourceGraph)
        mapping_run_time = time.time() - before_seconds
        print("  DFS Time:", mapping_run_time, end="   ")
        GM.mapping = {}
        for p, q, sim in list_of_mapped_preds:# [0]:  # read back the results
            a, b, c = q
            x, y, z = p
            GM.mapping[a] = x
            GM.mapping[b] = y
            GM.mapping[c] = z
            total_sim_score += sim
        GM.mapping['Total_Score'] = total_sim_score
        GM.mapping['Number_Mapped_Predicates'] = number_mapped_predicates
    elif algorithm == "ismags":
        s_grf, s_encoding, s_decoding = encode_graph_labels(sourceGraph)
        t_grf, t_encoding, t_decoding = encode_graph_labels(targetGraph)
        ismags = nx.isomorphism.ISMAGS(s_grf, t_grf)
        largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))  # False
        mapping_run_time = time.time() - before_seconds
        print(" ISMAGS Time:", mapping_run_time, end="  ")
        GM.mapping = largest_common_subgraph[0].copy()
        return largest_common_subgraph, s_decoding, t_encoding
    elif algorithm == "VF2":
        timeLimit = 30.0
        if __name__ == '__main__':
            # GM = isomorphvf2CB.MultiDiGraphMatcher(target_graph, source_graph)
            p1 = multiprocessing.Process(target=isomorphvf2CB.MultiDiGraphMatcher,
                     args=(target_graph, source_graph), name='MultiDiGraphMatcher')
            print(" VF2...", end="")
            p1.start()
            p1.join(timeout=timeLimit)
            p1.terminate()
        while p1.is_alive():
            print('.', end="")
            time.sleep(5)
        after_seconds = time.time()
        print("...VF2 ", after_seconds-before_seconds, end=" ")
        if p1.exitcode is None:     # a TimeOut
           numberOfTimeOuts += 1
           return 0
        res = GM.subgraph_is_isomorphic()
        for s,v,o in returnEdgesAsList(target_graph): # choose matching edge
            if s in GM.mapping and o in GM.mapping:
                z = returnEdgesBetweenTheseObjects(GM.mapping[s], GM.mapping[o], source_graph)
                if len(z) == 0:
                    continue
                elif v in z:  # identical v
                    list_of_mapped_preds.append([ [s,v,o],[GM.mapping[s], v, GM.mapping[o]] ])
                elif len(z) == 1:
                    list_of_mapped_preds.append([ [s, v, o], [GM.mapping[s], z[0], GM.mapping[o]] ])
                else:
                    tmp = find_most_similar_rel(v,z)
                    list_of_mapped_preds.append([ [s, v, o], [GM.mapping[s], tmp[0], GM.mapping[o]] ])
        return GM.mapping
    if len(GM.mapping) == 0:
        print(":-( NO Mapping:")
    else:
        if target_graph.number_of_nodes() >0:
           print("   ", len(GM.mapping), "mapped concept instances.",
                round(100 * len(GM.mapping) / target_graph.number_of_nodes(), 2),
                " %T  ", round(100 * len(GM.mapping) / source_graph.number_of_nodes(),2), " %S ")
        else:
            print(" Empty target graph ")


def find_most_similar_rel(t_reln, s_rel_list):
    rslt_list = []
    for z in s_rel_list:
        rslt_list.append(z + wn_sim_mine(t_reln, z, 'v'))
    best_match = sorted(rslt_list, key=lambda val: val[1], reverse=True)
    return best_match[0]


def develop_analogy(target_graph, source_graph):
    global GM
    global relationMapping
    global numberOfTimeOuts
    global list_of_mapped_preds
    global result_graph
    #GM.mapping.clear()
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

    result_graph = nx.MultiDiGraph()
    for pred1, pred2, scor in list_of_mapped_preds:
        result_graph.add_node(pred1[0]+"|"+pred2[0], label=pred1[0]+"|"+pred2[0])
        result_graph.add_node(pred1[2]+"|"+pred2[2], label=pred1[2]+"|"+pred2[2])
        result_graph.add_edge(pred1[0]+"|"+pred2[0], pred1[2]+"|"+pred2[2], label=pred1[1]+"|"+pred2[1])
    result_graph.graph['Graphid'] = target_graph.graph['Graphid']+"__"+source_graph.graph['Graphid']
    if False and len(list_of_mapped_preds) > 5:
        show_graph_in_FF(result_graph)

    calculate2Similarities(target_graph, source_graph, best_analogy)
    return 0


def return_analogy_result(target_graph, source_graph):
    global list_of_mapped_preds
    develop_analogy(target_graph, source_graph)
    return list_of_mapped_preds


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
            closest_sim = 0.0
            nearest_s_verb = "nULl"
            simmy = 0
            for s_verb in source_verbs:
                if s_verb:
                    if tRelation == s_verb:
                        simmy = 1
                    elif s2v.get_freq(head_word(tRelation) + '|VERB') is not None and \
                            s2v.get_freq(head_word(s_verb) + '|VERB') is not None:
                        simmy = s2v.similarity([head_word(tRelation) + '|VERB'], [head_word(s_verb) + '|VERB'])
                    #else:
                    #    print("**~ S2V", tRelation, s_verb, end="   ")
                if simmy >= closest_sim:
                    closest_sim = simmy
                    nearest_s_verb = s_verb
                    cached_predicate_mapping = [[tNoun1, tRelation, tNoun2, sNoun1, nearest_s_verb, sNoun2]]
            if not nearest_s_verb == 'nULl':
                rel_sim_total += closest_sim
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

def CSV_DEPRECATED():  # deprecated separate methods for calculating relation and conceptual similarity
    # return
    print("\nIn CSV() ", end="")
    global sourceGraph
    global targetGraph
    global CSVPath
    global CSVName
    global list_of_inferences
    with open(CSVPath + CSVName, 'a') as mappingFileAppendingData:  # Hmmm :-( Mapping results read from file
        with open(CSVPath + CSVName, 'r') as mappingFileReadingData:  # WHY read from mappingFileReadingData?
            LCSL = ""  # LCS Lin
            LCSW = ""  # LCS WuP
            verb1 = ""
            verb2 = ""
            max_value = 0.0  # Lin similarity
            max_value2 = 0.0  # WuP similarity
            mappingDataSourceItems = []
            mappingDataTargetItems = []
            subset1 = "none"
            subset2 = "none"
            rr = list(sourceGraph.nodes)
            rr2 = list(targetGraph.nodes)
            sourceGraphEdges = nx.get_edge_attributes(sourceGraph, 'label')  # Verb from sourceGraph
            targetGraphEdges = nx.get_edge_attributes(targetGraph, 'label')  # Verb from targetGraph
            x = 0
            y = 0
            d = 0
            e = 0

            inference = 0  # Noun/Concept metrics
            countLinZero = 0
            countWupZero = 0
            countNounList = 0
            nounLinSum = 0.0
            nounWupSum = 0.0
            countLinPerfect = 0.0
            countWupPerfect = 0.0
            averageLin = 0
            averageWup = 0

            num_target_nodes = targetGraph.number_of_nodes()  # Mapping Size
            target_nodes = targetGraph.nodes()  # list or something
            target_nodes_mapped = 0
            for x in target_nodes:
                if x in GM.mapping.keys():
                    target_nodes_mapped = + 1
            print("\nHI ", target_nodes_mapped, " of ", num_target_nodes, " nodes mapped.")
            concept_mapping_ratio = target_nodes_mapped / num_target_nodes

            countVerbList = 0  # Verb/Relation metrics
            countVerbLinZero = 0
            countVerbWupZero = 0
            verbsLinSum = 0.0
            verbsWupSum = 0.0
            verbLinPerfect = 0.0
            verbWupPerfect = 0.0
            averageVerbLin = 0
            averageVerbWup = 0

            AnalogicalSimilarity = 0.0

            LCSLlist = []
            filereader = csv.reader(mappingFileReadingData)
            fileWriter = csv.writer(mappingFileAppendingData, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
            mappingFileReadingData.readline()  # skip header
            for row in filereader:
                try:
                    mappingDataSourceItems.append(row[1])  # List of Source terms
                    mappingDataTargetItems.append(row[2])  # List of Target terms
                    if vbse > 3:
                        print(row[0], row[1], row[2], end="    ")
                except IndexError:
                    pass
            if vbse > 3:
                print("  \nData re-read ", end="")  # ##############################
            mappingFileReadingData.seek(1)  # ### Relational Similarity ####
            try:  # ##############################
                mappingFileReadingData.readline()  # skip header
                for map in filereader:  # cvsfile2
                    if vbse > 3:
                        print(map[1], end="  ")
                    # stop()
                    for y in range(len(mappingDataSourceItems)):
                        if vbse > 3:
                            print(" map[1]&friend ", map[1], mappingDataSourceItems[y], end=" ")
                        if sourceGraph.has_edge(" map[1],mappingDataSourceItems[y] ", map[1],
                                                mappingDataSourceItems[y]):  # find a Source edge
                            verb1 = sourceGraphEdges[map[1], mappingDataSourceItems[
                                y]]  # Extracts relation information between mapped terms by checking if both terms in a mapped pair have an edge relation to
                            print(" ok ")
                            verb1 = sourceGraphEdges[(map[1], mappingDataSourceItems[
                                y])]  # Extracts relation information between mapped terms by checking if both terms in a mapped pair have an edge relation to
                            # verb1 = (map[1],mappingDataSourceItems[y], sourceGraph)[0]
                        elif sourceGraph.has_edge(mappingDataSourceItems[y], map[1]):
                            # another mapped pair in their original graphs, and if so maps the verb/prepositional relation and performs Lin/WuP simialrity and
                            # verb1 = sourceGraphEdges[mappingDataSourceItems[y], map[1]]          #LCS checking as normal (As our graphs are directed, we have to check if terms have a realtion both ways manually)
                            # verb1 = sourceGraphEdges[(mappingDataSourceItems[y], map[1])]          #LCS checking as normal (As our graphs are directed, we have to check if terms have a realtion both ways manually)
                            verb1 = returnEdgesBetweenTheseObjects(mappingDataSourceItems[y], map[1], sourceGraph)[0]
                            # Commutativity Test for Verb
                            # if verb1 not in commutativeVerbList:
                            #    continue
                            #    #verb1 = None
                        if vbse > 3:
                            print("For now:", map[1], mappingDataSourceItems[y], verb1, end=" ")
                        # stop()
                        if targetGraph.has_edge(map[2], mappingDataTargetItems[y]):  # converse is 40 lines down
                            # verb2 = targetGraphEdges[(map[2],mappingDataTargetItems[y])]
                            verb2 = returnEdgesBetweenTheseObjects(mappingDataSourceItems[y], map[1], targetGraph)[0]
                            print("***VERB1&2", verb1, verb2, end=" ")
                            if verb1 == verb2:  # identical realtions
                                max_value = 1
                                max_value2 = 1
                                LCSW = verb1
                                LCSL = verb1
                            else:
                                syns1 = wn.synsets(verb2, pos='v')  # LIN verb similarity
                                syns2 = wn.synsets(verb1, pos='v')
                                for ss in syns1:
                                    for ss2 in syns2:
                                        d = ss.lin_similarity(ss2, semcor_ic)
                                        e = ss.wup_similarity(ss2)
                                        if d < 0.0000000001:
                                            d = 0
                                        if d > max_value:
                                            max_value = d
                                            LCSL = ss.lowest_common_hypernyms(ss2)
                                        if e > max_value2:
                                            max_value2 = e
                                            LCSW = ss.lowest_common_hypernyms(ss2)
                                        if d is None:
                                            d = 0
                                        if e is None:
                                            e = 0
                                        else:
                                            if d > max_value:
                                                max_value = d
                                            if e > max_value2:
                                                max_value2 = e
                                if max_value == 0:
                                    LCSL = "none"  # LCS Lin
                                if max_value2 == 0:
                                    LCSW = "none"  # LCS WuP
                                if max_value == 1:
                                    LCSL = verb1
                                if max_value2 == 1:
                                    LCSW = verb1

                            if (verb1 in GM.mapping) and (GM.mapping[verb1] == verb2):  # 29 Oct <<<<<<<<<<<<<<<<<<<<<<<
                                pass
                            else:
                                filewriter.writerow(['V4', verb1, verb2, max_value, max_value2,
                                                     simplifyLCSList(LCSL), simplifyLCSList(LCSW)])
                            e = 0
                            d = 0
                            max_value = 0.0
                            max_value2 = 0.0

                        elif (targetGraph.has_edge(mappingDataTargetItems[y],
                                                   map[2])):  # mappingDataTargetItems[y]<->x[2] 45 lines up
                            # verb2 = targetGraphEdges[(mappingDataTargetItems[y],map[2])]
                            verb2 = returnEdgesBetweenTheseObjects(mappingDataSourceItems[y], map[1], targetGraph)[0]
                            print("***VERB2", verb2)
                            if verb2 not in commutativeVerbList:
                                continue
                            if verb1 == verb2:
                                max_value = 1
                                max_value2 = 1
                                LCSW = verb1
                                LCSL = verb1
                            else:
                                syns1 = wn.synsets(verb2, pos='v')
                                syns2 = wn.synsets(verb1, pos='v')
                                for ss in syns1:
                                    for ss2 in syns2:
                                        d = ss.lin_similarity(ss2, semcor_ic)
                                        e = ss.wup_similarity(ss2)
                                        if d < 0.0000000001:
                                            d = 0
                                        if d > max_value:
                                            max_value = d
                                            LCSL = ss.lowest_common_hypernyms(ss2)
                                        if e > max_value2:
                                            max_value2 = e
                                            LCSW = ss.lowest_common_hypernyms(ss2)
                                        if d is None:
                                            d = 0
                                        if e is None:
                                            e = 0

                                        else:
                                            if d > max_value:
                                                max_value = d
                                            if e > max_value2:
                                                max_value2 = e
                                if max_value == 0:
                                    LCSL = "none"
                                if max_value2 == 0:
                                    LCSW = "none"
                                if max_value == 1:
                                    LCSL = verb1
                                if max_value2 == 1:
                                    LCSW = verb1
                                filewriter.writerow(['V5', verb1, verb2, max_value, max_value2,
                                                     simplifyLCSList(LCSL), simplifyLCSList(LCSW)])
                            e = 0
                            d = 0
                            max_value = 0.0
                            max_value2 = 0.0
            except IndexError:
                pass

            try:
                mappingFileReadingData.seek(0)
                # next(filereader)
                # Calculates average Lin/WuP simialrity for nouns   *** RE-READ ANALOGY FILE
                print("RE-re reading file ", end="  ")
                for line in filereader:
                    if vbse > 3:
                        print(1, end="")
                    if (line[0] == 'Type'):  # ? Skip the Header of the file
                        print("...CSV()...", end="")
                    elif (line[0][0][0] == 'N'):
                        nounLinSum += float(line[3])
                        nounWupSum += float(line[4])
                        countNounList += 1
                        if (float(line[3]) == 1):
                            countLinPerfect += 1
                        elif (float(line[3]) == 0):
                            countLinZero += 1
                        if (float(line[4]) == 1):
                            countWupPerfect += 1
                        elif (float(line[4]) == 0):
                            countWupZero += 1

                    elif (line[0][0] == 'V'):
                        verbsLinSum += float(line[3])
                        verbsWupSum += float(line[4])
                        countVerbList += 1
                        if (float(line[3]) == 1):
                            verbLinPerfect += 1
                        elif (float(line[3]) == 0):
                            countVerbLinZero += 1
                        if (float(line[4]) == 1):
                            verbWupPerfect += 1
                        elif (float(line[4]) == 0):
                            countVerbWupZero += 1
            except IndexError:
                pass

            # GROUNDED INFERENCES
            # generateCWSGInferences(sourceGraph, targetGraph)
            for a, r, b in list_of_inferences:
                filewriter.writerow(["Inference", a, r, b])
            # stop()
            # Write Merics
            filewriter.writerow(['#Noun Lin==0', countLinZero, 'of', countNounList])
            filewriter.writerow(['#Noun WuP==0', countWupZero, 'of', countNounList])
            if countNounList > 0:
                averageLin = nounLinSum / countNounList
                averageWup = nounWupSum / countNounList
            else:
                averageLin = 0
                averageWup = 0
            filewriter.writerow(['Average Noun Lin ', averageLin])
            filewriter.writerow(['Average Noun Wup ', averageWup])
            filewriter.writerow(['#Noun Lin== 1', countLinPerfect, 'of', countNounList])
            filewriter.writerow(['#Noun WuP== 1', countWupPerfect, 'of', countNounList])

            inferences = len(list_of_inferences)
            if inferences > 0:
                print("#### Inferences =", inferences)
            print("  countVerbList=", countVerbList)
            if countVerbList > 0:
                if (inferences == 0):
                    AnalogicalSimilarity = (0.5 * (verbsWupSum / countVerbList) + 0.2 * (nounWupSum / countNounList))
                    print("0 Infs, verbsWupSum countVerbList, nounWupSum: ",
                          round(verbsWupSum, 3), round(countVerbList, 3), round(nounWupSum, 3))
                else:
                    AnalogicalSimilarity = (
                                0.5 * (verbsWupSum / countVerbList) + 0.2 * (1 - (nounWupSum / countVerbList))
                                + 0.3 * (math.exp(-1 / inferences)))
                print("verbsWupSum countVerbList ", verbsWupSum, countVerbList)

                filewriter.writerow(['#Verb Lin==0', countVerbLinZero, 'of', countVerbList])
                filewriter.writerow(['#Verb Wup==0', countVerbWupZero, 'of', countVerbList])
                if countVerbList > 0:
                    averageVerbLin = (verbsLinSum / countVerbList)
                    averageVerbWup = (verbsWupSum / countVerbList)

                else:
                    averageVerbLin = 0
                    averageVerbWup = 0
                filewriter.writerow(['Average Verb Lin ', averageVerbLin])
                filewriter.writerow(['Average Verb Wup ', averageVerbWup])
                filewriter.writerow(['#Verb Lin== 1', verbLinPerfect, 'of', countVerbList])
                filewriter.writerow(['#Verb WuP== 1', verbWupPerfect, 'of', countVerbList])
                filewriter.writerow(['Analogical Similarity', AnalogicalSimilarity])
                print("#b Inference =", inferences, " ", list_of_inferences)
                print("ANALOGICAL Similarity", AnalogicalSimilarity, "   for",
                      sourceGraph.graph['Graphid'], targetGraph.graph['Graphid'])

            print(targetGraph.graph['Graphid'], sourceGraph.graph['Graphid'],
                  countLinZero, countWupZero, averageLin, averageWup, countLinPerfect, countWupPerfect,
                  countVerbLinZero, countVerbWupZero, averageVerbLin, averageVerbWup, verbLinPerfect, verbWupPerfect,
                  inferences, countNounList, countVerbList, AnalogicalSimilarity)
            writeSummaryFileData(targetGraph.graph['Graphid'], sourceGraph.graph['Graphid'],
                                 countLinZero, countWupZero, averageLin, averageWup, countLinPerfect, countWupPerfect,
                                 countVerbLinZero, countVerbWupZero, averageVerbLin, averageVerbWup, verbLinPerfect,
                                 verbWupPerfect,
                                 inferences, countNounList, countVerbList, AnalogicalSimilarity)  # mappedEdges
            print("Summary: ", round(AnalogicalSimilarity, 5), end="")
            mappingFileAppendingData.close()
            mappingFileReadingData.close()
            # stop()


# End CSV()


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
                if subjMap == None:  # Generate transferable symbol
                    subjMap = subj + " |INFER"
                if objMap == None:
                    objMap = obj + " |INFER"
                if relnMap == None:
                    relnMap = reln + " |INFER"
                print("#INFER(", subj, ", ", reln, ", ", obj,")  =>  (",subjMap, ", ", relnMap, ", ", objMap, end=")   ")
                list_of_inferences = list_of_inferences + [[subjMap, relnMap, objMap]]
    print("\n", end="")


def writeSummaryFileData_DEPRECATED(fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q):  # 18 params
    global CSVsummaryFileName
    global mapping_run_time
    with open(CSVsummaryFileName, "a+") as csvSummaryFileHandle:
        summaryFilewriter = csv.writer(csvSummaryFileHandle, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        summaryFilewriter.writerow(
            [fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q, mapping_run_time])

#("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "#Map Conc",
# "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "Infs", "#MapCon", "MapRels", "AnaSim")
def writeSummaryFileData(fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o, p ,q):  # 20 params
    global CSVsummaryFileName
    global mapping_run_time
    with open(CSVsummaryFileName, "a+") as csvSummaryFileHandle:
        summaryFilewriter = csv.writer(csvSummaryFileHandle, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        summaryFilewriter.writerow(
                    [fileName1, fileName2, a, b, c, d, e, f, g, h, i, j, l, m, n, o,
                     mapping_run_time, p, q])


# ###############################################################
# ############### Run Multiple Analogies ########################
# ###############################################################

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
    #GM = dict() #{}
    GM = isomorphvf2CB.MultiDiGraphMatcher(targetGraph, targetGraph)
    GM.mapping['Total_Score'] = 0
    GM.mapping['Number_Mapped_Predicates'] = 0
    # GM = isomorphvf2CB.DiGraphMatcher(targetGraph, targetGraph)
    # global sourceGraph
    global relationMapping
    relationMapping = []
    global numberOfTimeOuts
    numberOfTimeOuts = 0


undesirable_pair = ['Karla - Karla the Hawk.txt_v3.txt.dcorf__01 Antonietti - Four-Canals-Carocci.txt.dcorf.csv',
                    'Cord-Prob - Birthday Party.txt.dcorf__TumFort - The Identical Twins.txt_v3.txt.dcorf.csv',
                    'Cord-Prob - Birthday Party.txt.dcorf__TumFort - Tumor Problem.txt_v3.txt.dcorf.csv',
                    '94 Wharton - All Text.txt.dcorf__95 Keane - Relations B.txt.dcorf.csv',
                    '94 Wharton - All Text.txt.dcorf__95 Keane - Relations A.txt.dcorf.csv',
                    '87 Catrambone-B - Aquarium-Soln.txt.dcorf__Karla - Zerdia Literal Similarity.txt_v3.txt.dcorf.csv',
                    '87 Catrambone-B - Aquarium-Soln.txt.dcorf__Karla - Zerdia FOR match.txt.dcorf.csv',
                    '87 Catrambone-B - Aquarium-Soln.txt.dcorf__97 Markman - Fallsburg-Politic Sci-1.txt.dcorf.csv',
                    '87 Catrambone-B - Aquarium-Soln.txt.dcorf__95 Keane - Relations B.txt.dcorf.csv',
                    '84 Holyoak - Magic staff Source.txt.dcorf__Karla - Zerdia Literal Similarity.txt_v3.txt.dcorf.csv',
                    '84 Holyoak - Magic staff Source.txt.dcorf__Karla - Zerdia FOR match.txt.dcorf.csv',
                    '84 Holyoak - Magic staff Source.txt.dcorf__97 Markman - Gormond-CompSci Dept.txt.dcorf.csv',
                    '84 Holyoak - Magic staff Source.txt.dcorf__97 Markman - Fallsburg-Politic Sci-1.txt.dcorf.csv',
                    '87 Catrambone-A - Radiation-Problem-Dosage Version.txt.dcorf.T__87 Catrambone-B - Aquarium-Soln.txt.dcorf.T.csv',
                    '87 Catrambone-A - Radiation-Problem-Dosage Version.txt.dcorf.T__87 Catrambone-B - Aquarium-Soln.txt.dcorf.csv',
                    '87 Catrambone-C - Aquarium-Problem AND Solution.txt.dcorf__87 Catrambone-B - Aquarium-Soln.txt.dcorf.csv',
                    '87 Catrambone-A - Radiation-Problem-Dosage Version..txt.dcorf__87 Catrambone-B - Aquarium-Soln.txt.dcorf.csv']
#  not in ['Cord-Prob - Birthday Party.txt.dcorf.csv','94 Wharton - Document - Copy - Copy (2).txt.dcorf.csv',
#          '93 Gentner - Countries - Bolon Salan.txt.dcorf.csv']:

undesirable_target = ['Cord-Prob - Birthday Party.txt.dcorf.csv',
                      '87 Catrambone-A - Radiation-Problem-Dosage Version.txt.dcorf.csv',
                      '95 Keane - Attributes B.txt.dcorf.csv',
                      '95 Keane - Attributes A.txt.dcorf.csv',
                      '87 Catrambone-B - Aquarium-Soln.txt.dcorf.csv']

xyz = 'TumFort - Broken Lightbulb fragileglass Laser.txt_v3.txt.dcorf__TumFort - Broken Lightbulb fragileglass Laser.txt_v3.txt.dcorf.csv'


def blendWithAllSources(targetFile):
    global all_csv_files  # supply target, look for sources
    global targetGraph
    global sourceGraph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    global skip_over_previous_results
    if False:
        substring_location = targetFile.find('task')
        task = targetFile[substring_location:substring_location+5]
        these_csv_files = [i for i in all_csv_files if i[substring_location:substring_location+5] == task]
        # targetFileHead = targetFile[0:5]
    these_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith(targetFile[0:7])]
        # print("\n these_csv_files", targetFileHead, " ", these_csv_files, end=" with ")
        # all_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) ]
        # print(len(these_csv_files), " candidate sources.")
    targetGraph = build_graph_from_csv(targetFile).copy()
    targetGraph.graph['Graphid'] = targetFile
    if False:
        show_graph_in_FF(targetGraph)
        time.sleep(2.0)
        #input("Press Enter to continue...")
    p1 = targetFile.rfind(".")  # filetypeFilter
    # print("SUM: ", predicate_based_summary(targetGraph), end=" ")
    if targetGraph.number_of_edges() > max_graph_size:
        prune_peripheral_nodes(targetGraph)
    for nextSourceFile in these_csv_files:
    #for nextSourceFile in all_csv_files:
        # if nextSourceFile == targetFile: # skip self comparison
        #    continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile, "  <- ", nextSourceFile[0:p2], "=======")
        analogyFileName = targetFile[0:p1] + "__" + nextSourceFile[0:p2] + ".csv"
        if skip_over_previous_results and path.isfile(CSVPath + analogyFileName):
            print(" £skippy ", end="")
            continue
        elif analogyFileName in undesirable_pair:
            print(" £Undesirable pair ", end="")
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        sourceGraph = build_graph_from_csv(nextSourceFile).copy()
        sourceGraph.graph['Graphid'] = nextSourceFile
        if sourceGraph.number_of_edges() > max_graph_size:
            prune_peripheral_nodes(sourceGraph)
        elif sourceGraph.number_of_nodes() == 0:
            continue
        if False:
            #show_graph_in_FF(targetGraph)
            show_graph_in_FF(sourceGraph)
            time.sleep(3.0)
        develop_analogy(targetGraph, sourceGraph)
        # develop_analogy(sourceGraph, targetGraph)
        print("Source map:", return_ratio_of_mapped_source_predicates(sourceGraph),  # ...Source_Predicates
              "\tTarget map:", return_ratio_of_mapped_target_predicates(targetGraph))
        #if analogyCounter >= run_Limit:
        #   sys.exit()
        # stop()
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
        predicate_based_summary(targetGraph)
        if targetGraph.number_of_edges() > max_graph_size:
            # continue
            prune_peripheral_nodes(targetGraph)
        # show_graph_in_FF(sourceGraph)
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
    global algorithm
    analogyCounter = 0
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    writeSummaryFileData("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels", "#Map Preds",  "%Mapp",
                    "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "#Infs", "#MapCncpts",
                         "AvgRelSim", "LrgCmpnt", "#WekConnCpnt, Score", algorithm)
    # "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "#Infs", "#MapPreds", "AvgRelSim", "LargCpnnt", "#ConnCmpnnt", "Score")
    #targetFileHead = "orig_task" #targetFile[0:5]
    these_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith("93 Gentner")]
    # all_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith("93 Gentner")]
    #for next_target_file in these_csv_files:
    for next_target_file in all_csv_files:
        print("\n\n-----New Target: ", next_target_file, "------------------------------------------------------")
            # threeWordSummary(targetGraph)
            # 01 Antonietti - Four-Canals-Carocci.txt.dcorf.csv seems PROBLEMMATIC
        if next_target_file in undesirable_target:
            print(" £Undesirable target ")
            continue
        blendWithAllSources(next_target_file)  # next_target_file #one_csv_file[0]
        #blendWithAllSources(next_target_file)
        #deliberate_stop_early()
        #blendWithAllSources(next_target_file)
        # blendToPossibleTargets(next_target_file) # treat "target" as as a source
        analogyCounter += 1
    # csvSumryFil.close()
    # print(invalid_graphs, "of ",len(v)," Invalid graphs.  ","\n", analogyCounter, "analogies explored")



def blendWithSelectSources(targetFile):
    global all_csv_files  # supply target, look for sources
    global targetGraph
    global sourceGraph
    global analogyFileName
    global nextSourceFile
    global analogyCounter
    global max_graph_size
    global skip_over_previous_results
    if False:
        substring_location = targetFile.find('task')
        task = targetFile[substring_location:substring_location+5]
        these_csv_files = [i for i in all_csv_files if i[substring_location:substring_location+5] == task]
        # targetFileHead = targetFile[0:5]
    these_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith(targetFile[0:7])]
        # print("\n these_csv_files", targetFileHead, " ", these_csv_files, end=" with ")
        # all_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) ]
        # print(len(these_csv_files), " candidate sources.")
    targetGraph = build_graph_from_csv(targetFile).copy()
    targetGraph.graph['Graphid'] = targetFile
    if False:
        show_graph_in_FF(targetGraph)
        time.sleep(2.0)
        #input("Press Enter to continue...")
    p1 = targetFile.rfind(".")  # filetypeFilter
    # print("SUM: ", predicate_based_summary(targetGraph), end=" ")
    if targetGraph.number_of_edges() > max_graph_size:
        prune_peripheral_nodes(targetGraph)
    for nextSourceFile in these_csv_files:
    #for nextSourceFile in all_csv_files:
        # if nextSourceFile == targetFile: # skip self comparison
        #    continue
        p2 = nextSourceFile.rfind(".")
        print("\n\n#", mode, "================", "  ", targetFile, "  <- ", nextSourceFile[0:p2], "=======")
        analogyFileName = targetFile[0:p1] + "__" + nextSourceFile[0:p2] + ".csv"
        if skip_over_previous_results and path.isfile(CSVPath + analogyFileName):
            print(" £skippy ", end="")
            continue
        elif analogyFileName in undesirable_pair:
            print(" £Undesirable pair ", end="")
            continue
        resetAnalogyMetrics()
        temp_graph2.clear()
        sourceGraph = build_graph_from_csv(nextSourceFile).copy()
        sourceGraph.graph['Graphid'] = nextSourceFile
        if sourceGraph.number_of_edges() > max_graph_size:
            prune_peripheral_nodes(sourceGraph)
        if False:
            #show_graph_in_FF(targetGraph)
            show_graph_in_FF(sourceGraph)
            time.sleep(3.0)
        develop_analogy(targetGraph, sourceGraph)
        # develop_analogy(sourceGraph, targetGraph)
        print("Source map:", return_ratio_of_mapped_source_predicates(sourceGraph),  # ...Source_Predicates
              "\tTarget map:", return_ratio_of_mapped_target_predicates(targetGraph))
        analogyCounter += 1  # analogyFileName.close()
        # anyKey = input()


def blend_file_groups():
    global all_csv_files
    global analogyCounter
    global CSVPath
    global GM
    global csvSumryFil
    global algorithm
    analogyCounter = 0
    if not os.path.exists(CSVPath):
        os.makedirs(CSVPath)
    writeSummaryFileData("TARGET", "SOURCE", "#T Conc", "#T Rels", "#S Cons", "#S rels",  "#Map Preds",  "%Mapp",
                         "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel",  "#Infs", "#MapCon",
                         "AvgRelSim", "#ConnCpnnt, Score", algorithm)
    # "AvLin Con", "AvWu Con", "AvLin Rel", "AvWu Rel", "#Infs", "#MapPreds", "AvgRelSim", "LargCpnnt", "#ConnCmpnnt", "Score")
    target_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.endswith(".T.txt.dcorf.csv")]
    print(len(target_csv_files), target_csv_files)
    for next_target_file in target_csv_files:
    #for next_target_problem in all_csv_files:
        #group_csv_files = [i for i in all_csv_files if i.endswith(filetypeFilter) and i.startswith(next_target_file[0:7])]
        print("\n\n-----New Target: ", next_target_file, "------------------------------------------------------\n")
        if next_target_file in undesirable_target:
            print(" £Undesirable target ")
            continue
        blendWithSelectSources(next_target_file)  # next_target_file #one_csv_file[0]
        #blendWithAllSources(next_target_file)
        #deliberate_stop_early()
        #blendWithAllSources(next_target_file)
        # blendToPossibleTargets(next_target_file) # treat "target" as as a source
        analogyCounter += 1
        print("\n\n\n")
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
            if (cntr >= limit): #or (cntr % 5 == 0):  # delete nodes in batches of 5
                break
    print(grf.number_of_edges(), end="   ")


def predicate_based_summary(grf):
    edge_list = returnEdgesAsList(grf)
    result = []
    pr = nx.pagerank_numpy(grf)
    for s,v,o in edge_list:
        scor = grf.in_degree(s) + grf.out_degree(s) + grf.in_degree(o) + grf.out_degree(o)
        scor2 = pr[s] + pr[o]
        result.append([scor, scor2, s,v,o])
    best_analogy = sorted(result, key=lambda val: val[0], reverse=True)
    best_analogy2 = sorted(result, key=lambda val: val[1], reverse=True)
    # centroid of coincident relations
    if len(best_analogy) >0:
        return best_analogy[0]
    else:
        return []


# ######################################################################
# ##########################   GUI    ##################################
# ######################################################################

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
    #  addRelationsToMapping(sourceGraph, targetGraph)
    #  CSV()
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
    a = Button(window, text="Run Single Analogy", command=ok)
    # b = Button(window, text = "Map ALL sources",command = ok)
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
    buttonCommit = Button(window, height=1, width=10, text="Make Graph",  # ## input one
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
                         "Infs", "#MapCon", "MapRels", "AnaSim", "#Mapped Preds", "Score")
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

def show_pyvis():
    global all_csv_files
    targetGraph = build_graph_from_csv(all_csv_files[0])
    from pyvis.network import Network
    net = Network(notebook=True)
    net.from_nx(targetGraph)
    net.show("example.html")
    input("Press Enter to continue...")
#show_pyvis()


def test_build_graph():
    global all_csv_files
    print(len(all_csv_files), " candidate sources.")
    for targetFile in all_csv_files:
        print('-------------- ', targetFile, ' --------------- ', end="")
        before_seconds = time.time()
        #targetGraph = build_graph_from_csv(targetFile).copy()
        after_seconds = time.time()
        print(" %%% ", end="")
        #targetGraph.graph['Graphid'] = targetFile
        before_seconds2 = time.time()
        targetGraph2 = build_graph_from_csv_FAST(targetFile).copy()  # FAST
        after_seconds2 = time.time()
        targetGraph2.graph['Graphid'] = targetFile+"BB"
        # show_graph_in_FF(targetGraph)
        # show_graph_in_FF(targetGraph2)
        print(targetGraph.number_of_nodes(), after_seconds - before_seconds, " ",
              targetGraph2.number_of_nodes(), after_seconds2 - before_seconds2 )
        print(set( targetGraph.nodes() - targetGraph2.nodes() ))
        print(set( targetGraph2.nodes() - targetGraph.nodes() ))
        x=0

#test_build_graph()

print(is_proper_noun_DEPRECATED("Timmy Murphy"), is_proper_noun("Timmy Murphy"))
print(is_proper_noun_DEPRECATED("aardvark"), is_proper_noun("aardvark"))
print(is_proper_noun_DEPRECATED("Tom O'Sullivan"), is_proper_noun("Tom O'Sullivan"))
print(is_proper_noun_DEPRECATED("karla"), is_proper_noun("karla"))
print(is_proper_noun("Bezos"))
print(is_proper_noun("Timmy Murphy"))

x=['hot', 'roasting', 'angry', 'frustrated', 'enraged', 'cold', 'sad', 'relaxed', 'dejected']

def for_dermot():
    for attr1 in x:
        for attr2 in x:
            z = s2v.similarity(attr1 + '|ADJ', attr2 + '|ADJ')  # wn_sim_mine(attr1,attr2, 'ADJ')
            # s2v.similarity([head_word(tRelation) + '|VERB'], [head_word(s_verb) + '|VERB'])
            print(z, end=", ")
        print()
#for_dermot()


print(" S2V run-fly ", DFS.relational_distance("run", "fly"))
print(" S2V cown-duck ", DFS.conceptual_distance("cow", "duck"))
# stop()


loadConceptNet(CN_file)  # CN_dict
print("cvmn")
getCNConnections('four', '4')

blend_file_groups()
#blendAllFiles()