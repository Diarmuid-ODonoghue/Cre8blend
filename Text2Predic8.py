#############################
######## Text2Predic8 #######
#############################

import csv
import os
from os import path
import pprint
import re
import json
import sys

from nltk.tokenize import sent_tokenize
from nltk.tree import *
from functions import *
import nltk
from pycorenlp import StanfordCoreNLP
# move to Strava

#english_vocab = set(w.lower() for w in nltk.corpus.words.words())
java_path = "C:\Program Files\Java\jdk1.8.0_171\bin\java.exe"
os.environ['JAVAHOME'] = java_path

vbse = False  # VerBoSE mode for error reporting and tracing

basePath = "C:/Users/dodonoghue/Documents/Python-Me/data/"
localBranch = "iProva Texts 2/"# "test/"  #"iProvaData/"  
#localBranch = "test/"# "test/"  #"iProvaData/"
#localBranch = "Covid-19/"
#localBranch = "iProva/"
#localBranch = "MisTranslation data/"
localBranch = "Psychology data/"
#localBranch = "Covid-19 Publications Feb 21/covid_text/"
localBranch = "Sheffield-Plagiarism-Corpus/"
#localBranch = "Killians Summaries/"
#localBranch = "20 SIGGRAPH Abstracts - Stanford/"

#inPath = localPath + "SIGGRAPH Abstracts/" # "
#inPath = localPath + "psych texts/"
localPath = basePath  # localPath + inPath
#inPath = localPath + "psych data/"
inPath = localPath + localBranch  # "test/"
outPath = localPath + localBranch  

nlp = StanfordCoreNLP("http://localhost:9000")
###import stanza
###from stanza.server import CoreNLPClient
###stanza.download('en')
###nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
pp = pprint.PrettyPrinter(indent=4)

global document_triples
global sentence_triples
global sentence_triplesPP 
global set_of_raw_concepts

data = []
document_triples = []
sentence_triples = []
sentence_triplesPP = []
fullDocument = ""

global skip_over_previous_results
skip_over_previous_results = True

concept_tags = {'NN', 'NNS', 'PRP', 'PRP$'}  # NNP, NNPS
relation_tags = {'VB'}
illegal_concept_nodes = {"-RRB-", "-LRB-", "-RSB-" "-LSB-" "Unknown", "UNKNOWN", ",", ".", "?", "'s", "'", "''"}

def trim_concept_chain(text): # very long chains only, phrases
    if vbse > 3:
        print("   Trim c_c ", end="")
    "Extract nouns and Preps from coreferents that are entire phrases."
    str = nltk.word_tokenize(text.replace("_", " "))
    #print(" !!", str, end="!! ")
    tagged = nltk.pos_tag(str)
    ret = '_'.join([word for word, tag in tagged[:-1] if tag in concept_tags] + [tagged[-1][0]])
    if vbse > 3:
        print(" ->!!", ret, end="!! mirT ")
    return ret
#trim_concept_chain('cloth_captured_from_a_flapping_flag_it')


def processDocument(text): # full document
    global vbse
    global sentence_number
    global sentence_triples
    global sentence_triplesPP
    global set_of_raw_concepts
    list_of_sentences = sent_tokenize(text)
##    with CoreNLPClient(
##       annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, #milliseconds
##        memory='16G') as client:
##    ann = client.annotate(text)
    sents = sent_tokenize(text)
    output = nlp.annotate(text, properties={
    #    'annotators': 'tokenize, ssplit, parse, coref', 'outputFormat': 'json'  })
    #output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, parse, ner, dcoref',
        'outputFormat': 'json'})
    #output = nlp.annotate(text, properties={
    #     'annotators': 'tokenize, ssplit, parse, ner, coref',  # NEURAL coref
    #     'coref.algorithm': 'neural',
    #     'outputFormat': 'json'  })
    #output = nlp.annotate(text, properties={
    #     'annotators': 'tokenize, ssplit, pos, nerparse, lemma, nerparse, coref',  # NEURAL coref
    #     'coref.algorithm': 'neural',
    #     'outputFormat': 'json'  })
    #eg {'word': 'is', 'characterOffsetEnd': 1163, 'ner': 'O', 'after': ' ', 'lemma': 'be',
    # 'characterOffsetBegin': 1161, 'pos': 'VBZ', 'index': 9, 'originalText': 'is', #'before': ' ', 'speaker': 'PER0'},
    if output == "CoreNLP request timed out. Your document may be too long.":
        print("** Timeout of the Stanford Parser")
        list_of_sentences = []  # No parsed output to process
    elif isinstance(output, dict):
        try:
            coref = output['corefs']  ## OCCASIONALY DODGY RESULTS - ??? finds no corefs????
        except IndexError:
            coref = None
    elif type(output) is str: # or type(output) is unicode:
        output = json.loads(output, encoding='utf-8', strict=False)
        coref = output['corefs']
    else:
        print("** Stanford Parser Error - type:", type(output), end="")

    for i in range(len(list_of_sentences)):   # for each sentence #
        sentence_triples = []      #####################
        sentence_triplesPP = []
        sentence_number += 1
        if vbse > 3:
            print("\nSENT#", sentence_number, ":", sents[i].strip(), end="   ")
        if sents[i][0:14] == "CR Categories:":    # skip keyword list.
            break
        try:
            sent1 = output['sentences'][i]['parse']
        except IndexError:
            sent1 = None
        if sent1 is not None:
            try:
                sent2 = CoreferenceResolution(coref, sent1)
            except IndexError:
                sent2 = None

        tree = ParentedTree.fromstring(sent2)
        # tree.draw()  # show display

        Positions = getListPos(tree)

        Positions_depths = getOrderPos(Positions)
        Positions_leaves = getLeafPos(tree)
        # find the children of S
        # TODO implement new set of rule
        # locate all VP's in the sentence.
        posOfVP = findPosOfSpecificLabel(tree, "VP", Positions_depths, Positions_leaves)
        #print("***Position of VP**** ", posOfVP, "*****")

        ####################
        ######  VP  ########
        ####################
        if posOfVP == None:
            if vbse > 3:
                print("Gotcha no VP")  
        else: 
            for z in posOfVP:   # iterative over VP's
                Triple = []
                Verb = ""
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                if vbse > 3:
                    print("child ", child)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "VP":
                        if vbse > 3:
                            print("True VP1 ")
                        NextStep = False
                        # break out and stop working with this VP
                    else:
                        if vbse > 10:
                            print("non-VP1 ", end="")
                ###########################
                # If next step still equals true then there is no VP child of the current VP and
                # we can procede to the next step.
                if NextStep:
                    VerbTree = child[0]
                    Verb = child[0]
                    Verb = findLeavesFromNode(PositionInTree(Verb, Positions_depths), Positions_leaves)
                    Verb = checkLabelLeaf(tree, Verb)
                    if vbse > 10:
                        print("Verb:", Verb)
                    Subject = "Unknown"

                    LeftSibling = findLeftSiblingCurrentLevel(z, Positions_depths)
                    LeftSiblingPos = PositionInTree(LeftSibling, Positions_depths)
                    # print(checkLabel(tree, LeftSiblingPos))
                    RunCheck = True
                    try:
                        LeftSiblingLabel = checkLabel(tree, LeftSiblingPos)
                        if vbse > 3:
                            print("Try left-sibling ", end=" ")
                    except:
                        RunCheck = False
                        if vbse > 3:
                            print("No left-sibling", end=" ")

                    if RunCheck and LeftSiblingLabel == "NP":
                        leaves = findLeavesFromNode(LeftSiblingPos, Positions_leaves)
                        #print(leaves)
                        Subject = leaves[len(leaves) - 1]
                        #print(Subject)
                        Subject = checkLabelLeaf(tree, Subject)
                        if vbse > 3:
                            print("true NP1, leavels", leaves, "Subject:", Subject)
                        #print(Subject)
                    else:
                        # If left sibling isnt a NP then check parent and its NP, repeat until you find NP.
                        CurrentVP = z  # this will change later to x or something when I loop
                                       # through all of the VP's
                        cont = True
                        counter = 0   #why?

                        while cont == True and counter<10:
                            counter+=1
                            # get parent of this VP
                            Parent = findParentNode(PositionInTree(CurrentVP, Positions_depths), Positions_depths)
                            if vbse > 3:
                                print("?????????????Parent", Parent, "???????????????")
                            # print(CurrentVP)
                            # print(Parent)
                            if vbse > 3:
                                print("This is the parent node")
                            # now that we have parent, check its leftsibling
                            ParentLeftSibling = findLeftSiblingCurrentLevel(Parent, Positions_depths)
                            # print(ParentLeftSibling)
                            # now check the label of the parents left sibling, if it is an NP then use the above code, if it is node then repeat the process
                            ParentLeftSiblingPOS = PositionInTree(ParentLeftSibling, Positions_depths)
                            RunCheck = True
                            try:
                                ParentLeftSiblingPOSLabel = checkLabel(tree, ParentLeftSiblingPOS)
                            except:
                                RunCheck = False

                            if RunCheck and ParentLeftSiblingPOSLabel == "NP":
                                if vbse > 3:
                                    print("trueNP2 ")
                                leaves = findLeavesFromNode(ParentLeftSiblingPOS, Positions_leaves)
                                if vbse > 3:
                                    print("Leaves", leaves)
                                Subject = leaves[len(leaves) - 1]
                                if vbse > 3:
                                    print("Subject", Subject)
                                Subject = checkLabelLeaf(tree, Subject)
                                if vbse > 3:
                                    print("Subject", Subject)
                                cont = False
                                break
                            else:
                                CurrentVP = Parent

                    # now that I have the subject and Verb I should combine these together and create a double.
                    if Subject.count("_")>=2:
                        #print(Subject,"->", end="")
                        Subject = trim_concept_chain(Subject)
                        #print(Subject,"   ", end="")
                    Triple.append(Subject)

                    Triple.append(Verb)
                    if vbse > 3:
                        print("*************Partial Triple", Triple)

                    # now locate the OBJECT - if there is one.
                    Obj = "Unknown-Obj "
                    # reuse some of the code from previous rule to find closest NP on the right of the verb.
                    ListOfNP = findPosOfSpecificLabel(tree, "NP", Positions_depths, Positions_leaves)
                    PosOfVerbTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    if vbse >3:
                        print(PosOfVerbTree)
                        print(ListOfNP)
                    index = []
                    if ListOfNP:              #dod
                        for x in ListOfNP:
                            index.append(Positions.index(Positions_depths[x[0]][x[1]]))

                    if vbse > 3:
                        print(index)
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosOfVerbTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closest has an NP child, if it does work from this node instead
                    loop = True
                    count = 0
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)
                    if vbse >3:
                        print("currentNode", currentNode)
                    while loop and count<10:
                        currentNodePOS = PositionInTree(currentNode,Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
    
                        if(currentNodeChildTreePOS in Positions_leaves):
                            loop = False
                            break
                        elif checkLabel(tree, currentNodeChildTreePOS)=="NP":
                            currentNode = currentNodeChild[0]
                        else:
                            leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                            Obj = checkLabelLeaf(tree, leaves[len(leaves)-1])
                            loop = False
                            break
                    if vbse >3:
                        print("Obj=", Obj, " ")

                    if Obj!= ".":
                        if Obj.count("_")>=2:         # trim coreference Phrases
                            #print(Obj,"=>", end="")
                            Obj = trim_concept_chain(Obj)
                            #print(Obj, end="   ")
                        Triple.append(Obj)
                        #print(" TRIPLE: ", end="")
                        #print(Triple, end="")
                        sentence_triples.append(Triple)  # end PosOfVP for loop

        ####################
        ######  PP  ########
        ####################

        PosOfPP = findPosOfSpecificLabel(tree, "PP", Positions_depths, Positions_leaves)
        if vbse: print("$$$$$$$ PosOfPP", PosOfPP, "$$$$$$$")
        #global sentence_triplesPP
        # sentence_triplesPP = []
        if PosOfPP is None:
            if vbse:
                print("No PP found")
        else: 
            if vbse: print("posOfPP", PosOfPP)
            for z in PosOfPP:
                Triple = []
                Preposition = ""
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "PP":
                        if vbse: print("CheckLabel is True")
                        NextStep = False
                    else:
                        if vbse: print("CheckLabel is False")

                if NextStep:
                    Preposition = child[0]
                    Preposition = findLeavesFromNode(PositionInTree(Preposition, Positions_depths), Positions_leaves)
                    if vbse:
                        print(tree)
                        print(Preposition)
                    if vbse: print("Preposition index:", Preposition)
                    
                    if vbse: print("Preposition", Preposition)
                    if type(Preposition) == list:
                        if len(Preposition[0]) >1 :           ## ERROR from here
                            Preposition = [Preposition[0]]
                    if vbse: print("2 tree[Preposition]", tree[Preposition])
                    
                    Preposition = checkLabelLeaf(tree, Preposition) 

                    # find NP on the left
                    PosPPTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    closest = 0
                    currentDif = -1000000
                    if vbse:
                        print("index", index)
                        print("PosPPTree", PosPPTree)

                    """if isinstance(index, int):  # ulgy hack
                        index = [index]
                        #print("is int")   """

                    try:
                        index
                    except NameError:  # no Verb
                        index = []

                    for y in index:   # position of V
                        diff = y - PosPPTree
                        if (diff < 0 and diff > currentDif):
                            currentDif = diff
                            closest = y
                    # now that you have closest NP get the children
                    leaves = findLeavesFromNode(Positions[closest], Positions_leaves)
                    # add the right most leaf to the triple
                    leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                    Triple.append(leafLabel)
                    Triple.append(Preposition)

                    # now get NP on the right
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosPPTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closet has an NP child, if it does work from child
                    leafLabel = "UNKNOWN"
                    loop = True
                    count = 0
                    if vbse:
                        print(" closest=", closest, "Positions", Positions, len(Positions), end=" ")

                    if closest>= len(Positions): closest=(len(Positions)-1) ## No NP
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)
                    if vbse: print("currentNode:", currentNode)

                    while (currentNode != None) and (loop and count<10):         # Why 10? 10 attempts?
                        # check if child is a leaf node first
                        # ClosestPosInOrderList = findPosInOrderList(Positions[closest],Positions_depths)
                        # childOfClosest = findChildNodes(Positions[closest], ClosestPosInOrderList, Positions_depths)
                        # childOfClosestTreePOS = PositionInTree(childOfClosest[0], Positions_depths)
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        if currentNodeChild == []:
                            pass
                        else: 
                            currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
                            if (currentNodeChildTreePOS in Positions_leaves):
                                loop = False
                                break
                            elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                                currentNode = currentNodeChild[0]
                            else:
                                leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                                leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                                loop = False
                                break
                        count+=1

                    Triple.append(leafLabel)
                    sentence_triplesPP.append(Triple)

        # *********************************
        # *        POST PROCESSING        *
        # *********************************

        # TODO post processing
        x=0
        sentence_triples_copy = sentence_triples.copy()
        sentence_triplesPP_copy = sentence_triplesPP.copy()
        for x in sentence_triples_copy:
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triples.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triples.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triples.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triples.remove(x)

        for x in sentence_triplesPP_copy:  # remove invalid triples
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triplesPP.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triplesPP.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triplesPP.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triplesPP.remove(x)
        x = 0
        for vb_triple in sentence_triples:  ## Phrasal verb composition
            for prp_triple in sentence_triplesPP:  # X vb Y followed by  X prp Y in same sentence
                if (prp_triple[0]== vb_triple[0] and prp_triple[2]== vb_triple[2] and
                    (vb_triple[1] +" " +prp_triple[1]) in sents[i]): # sequence bv prp in the text
                    if prp_triple in sentence_triplesPP:       #already removed?
                        sentence_triplesPP.remove(prp_triple)
                    if vb_triple in sentence_triples:
                        sentence_triples.remove(vb_triple)
                    sentence_triples.append([prp_triple[0], vb_triple[1]+"_"+prp_triple[1], prp_triple[2]])
                    print(vb_triple[1]+"_"+prp_triple[1], end="")

        print("\nsentence_triples: ", sentence_triples)
        print("sentence_triplesPP: ", sentence_triplesPP, end="  ")
        print()
        #tree.draw()  # show display parse tree
        
        document_triples.append(sentence_triples)
        document_triples.append(sentence_triplesPP)

    return


def process_sentence(text):  # full document
    global sentence_triples
    global sentence_triplesPP
    global set_of_raw_concepts
    list_of_sentences = sent_tokenize(text)
    sents = sent_tokenize(text)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, parse, ner, dcoref',
        'outputFormat': 'json'})
    if output == "CoreNLP request timed out. Your document may be too long.":
        print("** Timeout of the Stanford Parser")
        list_of_sentences = []  # No parsed output to process
    elif isinstance(output, dict):
        try:
            coref = output['corefs']  ## OCCASIONALY DODGY RESULTS - ??? finds no corefs????
        except IndexError:
            coref = None
    elif type(output) is str:  # or type(output) is unicode:
        output = json.loads(output, encoding='utf-8', strict=False)
        coref = output['corefs']
    else:
        print("** Stanford Parser Error - type:", type(output), end="")

    for i in range(len(list_of_sentences)):
        sentence_triples = []
        sentence_triplesPP = []
        if vbse >3:
            print("\nSENT", sents[i].strip(), end="   ")
        try:
            sent1 = output['sentences'][i]['parse']
        except IndexError:
            sent1 = None
        if sent1 is not None:
            try:
                sent2 = CoreferenceResolution(coref, sent1)
            except IndexError:
                sent2 = None

        tree = ParentedTree.fromstring(sent2)
        # tree.draw()  # show display

        Positions = getListPos(tree)

        Positions_depths = getOrderPos(Positions)
        Positions_leaves = getLeafPos(tree)
        # find the children of S
        # TODO implement new set of rule
        # locate all VP's in the sentence.
        posOfVP = findPosOfSpecificLabel(tree, "VP", Positions_depths, Positions_leaves) # Position of VP

        ####################
        ######  VP  ########
        ####################
        if posOfVP == None:
            if vbse >3:
                print("Gotcha no VP")
        else:
            for z in posOfVP:  # iterative over VP's
                Triple = []
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "VP":
                        NextStep = False  # break out and stop working with this VP
                ###########################
                # If no VP child of the current VP then we can proceede to the next step.
                if NextStep:
                    VerbTree = child[0]
                    Verb = child[0]
                    Verb = findLeavesFromNode(PositionInTree(Verb, Positions_depths), Positions_leaves)
                    Verb = checkLabelLeaf(tree, Verb)
                    Subject = "Unknown"

                    LeftSibling = findLeftSiblingCurrentLevel(z, Positions_depths)
                    LeftSiblingPos = PositionInTree(LeftSibling, Positions_depths)
                    # print(checkLabel(tree, LeftSiblingPos))
                    RunCheck = True
                    try:
                        LeftSiblingLabel = checkLabel(tree, LeftSiblingPos)
                    except:
                        RunCheck = False  # No left-sibling

                    if RunCheck and LeftSiblingLabel == "NP":
                        leaves = findLeavesFromNode(LeftSiblingPos, Positions_leaves)
                        Subject = leaves[len(leaves) - 1]
                        Subject = checkLabelLeaf(tree, Subject)

                    else:  # If left sibling isnt a NP then check parent and its NP, repeat until you find NP.
                        CurrentVP = z  # this will change later to x or something when I loop over all the VP's
                        cont = True
                        counter = 0  # why?

                        while cont == True and counter < 10:
                            counter += 1
                            # get parent of this VP
                            Parent = findParentNode(PositionInTree(CurrentVP, Positions_depths), Positions_depths)
                            # print(CurrentVP)
                            # Parent is the parent node, we have parent  check its leftsibling
                            ParentLeftSibling = findLeftSiblingCurrentLevel(Parent, Positions_depths)
                            # print(ParentLeftSibling)
                            # now check the label of the parents left sibling, if it is an NP then use the above code, if it is node then repeat the process
                            ParentLeftSiblingPOS = PositionInTree(ParentLeftSibling, Positions_depths)
                            RunCheck = True
                            try:
                                ParentLeftSiblingPOSLabel = checkLabel(tree, ParentLeftSiblingPOS)
                            except:
                                RunCheck = False

                            if RunCheck and ParentLeftSiblingPOSLabel == "NP": # trueNP2
                                leaves = findLeavesFromNode(ParentLeftSiblingPOS, Positions_leaves)
                                Subject = leaves[len(leaves) - 1]
                                Subject = checkLabelLeaf(tree, Subject)
                                cont = False
                                break
                            else:
                                CurrentVP = Parent

                    # now that I have the subject and Verb I should combine these together and create a double.
                    if Subject.count("_") >= 2:
                        # print(Subject,"->", end="")
                        Subject = trim_concept_chain(Subject)
                        # print(Subject,"   ", end="")
                    Triple.append(Subject)

                    Triple.append(Verb)  # Partial Triple

                    # now locate the OBJECT - if there is one.
                    Obj = "Unknown-Obj "
                    # reuse some of the code from previous rule to find closest NP on the right of the verb.
                    ListOfNP = findPosOfSpecificLabel(tree, "NP", Positions_depths, Positions_leaves)
                    PosOfVerbTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])

                    index = []
                    if ListOfNP:  # dod
                        for x in ListOfNP:
                            index.append(Positions.index(Positions_depths[x[0]][x[1]]))

                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosOfVerbTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closest has an NP child, if it does work from this node instead
                    loop = True
                    count = 0
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)
                    while loop and count < 10:
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)

                        if (currentNodeChildTreePOS in Positions_leaves):
                            loop = False
                            break
                        elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                            currentNode = currentNodeChild[0]
                        else:
                            leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                            Obj = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                            loop = False
                            break

                    if Obj != ".":
                        if Obj.count("_") >= 2:  # trim coreference Phrases
                            Obj = trim_concept_chain(Obj)
                        Triple.append(Obj)
                        # print(" TRIPLE: ", Triple, end="")
                        sentence_triples.append(Triple)  # end PosOfVP for loop

        ####################
        ######  PP  ########
        ####################

        PosOfPP = findPosOfSpecificLabel(tree, "PP", Positions_depths, Positions_leaves)
        if vbse: print("$$$$$$$ PosOfPP", PosOfPP, "$$$$$$$")
        # global sentence_triplesPP
        # sentence_triplesPP = []
        if PosOfPP is None:
            if vbse:
                print("No PP found")
        else:  # posOfPP
            for z in PosOfPP:
                Triple = []
                Preposition = ""
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "PP":
                        NextStep = False  # CheckLabel is True"
                    # CheckLabel is False

                if NextStep:
                    Preposition = child[0]
                    Preposition = findLeavesFromNode(PositionInTree(Preposition, Positions_depths), Positions_leaves)
                    # Preposition index:", Preposition)

                    if type(Preposition) == list:
                        if len(Preposition[0]) > 1:  ## ERROR from here
                            Preposition = [Preposition[0]]
                    # tree[Preposition]", tree[Preposition])

                    Preposition = checkLabelLeaf(tree, Preposition)

                    # find NP on the left
                    PosPPTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    closest = 0
                    currentDif = -1000000

                    """if isinstance(index, int):  # ulgy hack
                        index = [index]
                        #print("is int")   """

                    try:
                        index
                    except NameError:  # no Verb
                        index = []

                    for y in index:  # position of V
                        diff = y - PosPPTree
                        if (diff < 0 and diff > currentDif):
                            currentDif = diff
                            closest = y
                    # now that you have closest NP get the children
                    leaves = findLeavesFromNode(Positions[closest], Positions_leaves)
                    # add the right most leaf to the triple
                    leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                    Triple.append(leafLabel)
                    Triple.append(Preposition)

                    # now get NP on the right
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosPPTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closet has an NP child, if it does work from child
                    leafLabel = "UNKNOWN"
                    loop = True
                    count = 0

                    if closest >= len(Positions): closest = (len(Positions) - 1)  ## No NP
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)

                    while (currentNode != None) and (loop and count < 10):  # Why 10? 10 attempts?
                        # check if child is a leaf node first
                        # ClosestPosInOrderList = findPosInOrderList(Positions[closest],Positions_depths)
                        # childOfClosest = findChildNodes(Positions[closest], ClosestPosInOrderList, Positions_depths)
                        # childOfClosestTreePOS = PositionInTree(childOfClosest[0], Positions_depths)
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        if currentNodeChild == []:
                            pass
                        else:
                            currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
                            if (currentNodeChildTreePOS in Positions_leaves):
                                loop = False
                                break
                            elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                                currentNode = currentNodeChild[0]
                            else:
                                leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                                leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                                loop = False
                                break
                        count += 1

                    Triple.append(leafLabel)
                    sentence_triplesPP.append(Triple)

        # *********************************
        # *        POST PROCESSING        *
        # *********************************

        # TODO post processing
        x = 0
        sentence_triples_copy = sentence_triples.copy()
        sentence_triplesPP_copy = sentence_triplesPP.copy()
        for x in sentence_triples_copy:
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triples.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triples.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triples.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triples.remove(x)

        for x in sentence_triplesPP_copy:  # remove invalid triples
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triplesPP.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triplesPP.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triplesPP.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triplesPP.remove(x)
        x = 0
        for vb_triple in sentence_triples:  ## Phrasal verb composition
            for prp_triple in sentence_triplesPP:  # X vb Y followed by  X prp Y in same sentence
                if (prp_triple[0] == vb_triple[0] and prp_triple[2] == vb_triple[2] and
                        (vb_triple[1] + " " + prp_triple[1]) in sents[i]):  # sequence bv prp in the text
                    if prp_triple in sentence_triplesPP:  # already removed?
                        sentence_triplesPP.remove(prp_triple)
                    if vb_triple in sentence_triples:
                        sentence_triples.remove(vb_triple)
                    sentence_triples.append([prp_triple[0], vb_triple[1] + "_" + prp_triple[1], prp_triple[2]])
                    print(vb_triple[1] + "_" + prp_triple[1])

        #print("\nsentence_triples: ", sentence_triples)
        #print("sentence_triplesPP: ", sentence_triplesPP, end="  ")
        #print()
        # tree.draw()  # show display parse tree

        document_triples.append(sentence_triples)
        document_triples.append(sentence_triplesPP)

    return


# *********************************
# *   OUTPUT PREPARATION        *
# *********************************

def generate_output_CSV_file(fileName):
    global document_triples
    testList = BringListDown1D(document_triples)
    if vbse >3:
        print(testList)
    heading = [["NOUN", "VERB/PREP", "NOUN"]]
    with open(outPath + fileName+".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, lineterminator='\n')
        write.writerows(heading)
        write.writerows(testList)
    resultFile.close()
    return


def processAllTextFiles():
    global inPath
    global document_triples
    global sentence_number
    global set_of_raw_concepts
    global skip_over_previous_results
    fileList = os.listdir(inPath)
    txt_files = [i for i in fileList if i.endswith('.txt')]
    txt_files.remove('Cache.txt')  # System file for Text2Predic8
    csv_files = [i for i in fileList if i.endswith('.csv')]
    for fileName in txt_files:
        set_of_raw_concepts = set()
        sentence_number = 0
        print("\n####################################################")
        print("FILE ", inPath, "&&", fileName)
        if skip_over_previous_results and path.isfile(inPath + fileName + ".dcorf.csv"):
            print(" £skippy ", end="")
            continue
        global data
        data = []
        document_triples = []
        try:
            file = open(inPath+fileName, "r", encoding="utf8", errors='replace')
        except Exception as err:
             print("Erro {}".format(err))
        full_document = file.read()
        full_document_list = full_document.split()
        text_chunk_size = 500
        text_chunk_start = 0
        text_chunk_end = text_chunk_size
        while text_chunk_start < len(full_document_list):
            z = full_document_list[text_chunk_start:text_chunk_end]
            documentSegment = " ".join(z)
            processDocument(documentSegment)
            text_chunk_start += text_chunk_size -2
            text_chunk_end = text_chunk_end + text_chunk_size
            #if text_chunk_start >=4500:
            #    print("End Of Document Truncated", end="")
            if text_chunk_end > len(full_document_list):
                text_chunk_end = min(len(full_document_list),len(full_document_list))
            else:
                if text_chunk_start < len(full_document_list):
                    while (text_chunk_end >= text_chunk_start + (text_chunk_size - 100)) and \
                        ("." not in full_document_list[text_chunk_end] or \
                        "?" not in full_document_list[text_chunk_end] or \
                        ":" not in full_document_list[text_chunk_end]):  # split chunks @ full-stop, where reasonable
                        text_chunk_end -= 1
        generate_output_CSV_file(fileName) # uses documentSegment, documentTriples
        set_of_unique_concepts = set()
        for l in document_triples:
            for a,b,c in l:
                set_of_unique_concepts.add(a)
                set_of_unique_concepts.add(c)
        print("CONCEPT COUNT", ",", fileName, ",", len(set_of_unique_concepts))
    return


def add_line_to_output_CSV_file(fileName):
    global document_triples
    testList = BringListDown1D(document_triples)
    with open(outPath + fileName+".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, lineterminator='\n')
        write.writerows(heading)
        write.writerows(testList)
    resultFile.close()
    return


def process_microsoft_paraphrase_corpus():
    global inPath
    global data
    global document_triples
    global sentence_number
    global set_of_raw_concepts
    global skip_over_previous_results
    heading = [["NOUN", "VERB/PREP", "NOUN"]]
    data = []
    document_triples = []
    #fileList = os.listdir(inPath)
    basePath = "C:/Users/dodonoghue/Documents/Python-Me/data/Microsoft Paraphrase Corpus/"
    in_file = basePath + 'msr_paraphrase_test.txt' # [i for i in fileList if i.endswith('msr_paraphrase_test.txt')]
    with open(in_file +".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        write.writerows(heading)
        with open(in_file, 'r', encoding='utf8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            try:
                for row in csvreader:
                    # print(row)
                    if len(row) < 5:
                        print("ERROR! ", row)
                        continue
                        # sys.exit("data error in ", row)
                    document_triples = []
                    process_sentence(row[3])
                    list_a = BringListDown1D(document_triples).copy()
                    document_triples = []
                    process_sentence(row[4])
                    list_b =  document_triples.copy() #BringListDown1D(document_triples)
                    # write.writerows([[row[0], '\t', row[1], '\t', row[2], '\t', list_a, '\t', list_b]])
                    write.writerows([[row[0], row[1], row[2], list_a , list_b]])
                    write.writerows([])
                    #print()
            except csv.Error as e:
                print("howya", row)
                sys.exit('error', e)
        #generate_output_CSV_file(fileName) # uses documentSegment, documentTriples
        #write.writerows(testList)
    resultFile.close()
    return
#csv.writer(analogyFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

vbse = 0

print("Type   processAllTextFiles()   to generate graphs from ", inPath)
# processAllTextFiles()
process_microsoft_paraphrase_corpus()
