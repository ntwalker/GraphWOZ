import torch
import json
import time
import random
import numpy as np
from collections import defaultdict
from load_dialogue_data import extract_triples, create_mappings
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
from os import listdir
import os.path as osp
from os.path import isfile, join

import torch_geometric.transforms as T
from torch.functional import F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from homogeneous_graph_transform import json_to_graph_indices, encode_graphs
from gnn_dialogue_homogeneous import DialogueHomogeneousDataset

# BASIC COMPARISON BETWEEN MENTION AND ENTITIES, COSINE SIMILARITY
# 

def entity_cosine_similarity(data):

    target_scores = dict()
    correct = 0

    predictions = []

    mrr = 0

    for pair in data.mention_edges:
        selected_target = -1 # Index of selected target for source (mention) node
        highest_similarity = 0 # Keep track of the highest cosine similarity

        source = pair[0]
        true_target = pair[1]

        target_scores[source] = dict() #Save similarities

        for i, node in enumerate(data.x):

            if i == source:
                continue # Don't compare mention node to itself

            shortened_x = torch.Tensor(data.x[:,:,:,0]) #Ignore the last dimension
            #mask = data.x_mask.unsqueeze(-1).unsqueeze(-1).expand(data.x.size()).float() #Works for original data but disgustingly slow
            #mask = data.x_mask.unsqueeze(-1).expand(data.x.size()).float() #Original, breaks with type dimension added
            mask = data.x_mask.unsqueeze(-1).expand(shortened_x.size()).float() # New with ignoring type dimension
            #masked_data = data.x * mask # Mask padding tokens across the data
            masked_data = shortened_x * mask # Mask padding tokens across the shortened data (type dimension ignored)

            #Original method (without attention mask, averaging WITH padding, not ideal)
            #source_avg = data.x[source].mean(axis=0)
            #target_avg = node.mean(axis=0)

            source_avg = masked_data[source].mean(axis=0)
            target_avg = masked_data[i].mean(axis=0)

            sim = torch.cosine_similarity(source_avg.reshape(1,-1), target_avg.reshape(1,-1))
            target_scores[source][i] = sim.detach()

            if sim > highest_similarity:
                selected_target = i
                highest_similarity = sim

        if selected_target == true_target:
            correct += 1

        pred = (data.x_labels[source], data.x_labels[selected_target], data.x_labels[true_target])
        predictions.append(pred)

        ranked_entity_matches = [el[0] for el in sorted(target_scores[source].items(), key=lambda x: x[1], reverse=True)] #Most to least likely to be a match

        for i, m, in enumerate(ranked_entity_matches):
            if m == true_target:
                mrr += (1 / (i+1))

        #print("Highest scoring link: %d to %d" %(source, selected_target))
        #print("Predicted: ", data.x_labels[source], "->", data.x_labels[selected_target], "Score: %f" %highest_similarity)
        #print("True link: %d to %d" %(source, true_target))
        #print("True: ", data.x_labels[source], "->", data.x_labels[true_target])
        #print("\n")

    return target_scores, correct, predictions, mrr

# Select entity mention to entity link based on cosine similarity of average BERT vectors of the two nodes

def nongraph_entity_link(data_list, file_prefix):
    total = 0
    true_positives = 0
    global_predictions = []
    global_mrr = 0
    for graph in data_list:
        if len(graph.mention_edges) == 0:
            continue
        target_scores, correct, predictions, mrr_score = entity_cosine_similarity(graph)
        true_positives += correct
        global_mrr += mrr_score
        #print(sorted(target_scores.items(), key=lambda x: x[1], reverse=True))
        for mention, scores in target_scores.items():
            #print("Top scoring entities for mention index %d:" %mention)
            ranked_entity_matches = [el[0] for el in sorted(scores.items(), key=lambda x: x[1], reverse=True)] #Most to least likely to be a match
            
            #print("Top Match:")
            #print(graph.x_labels[mention], graph.x_labels[ranked_entity_matches[0]])
            total += 1

        print("Current MRR: %f" %((1/total) * global_mrr))
        global_predictions.extend(predictions)
    global_mrr = (1/len(total)) * global_mrr
    with open(file_prefix + "_cosine_predictions.txt", "w") as file:
        file.write("Mention, Predicted Referent Entity, True Referent Entity")
        file.write("\n-------------\n")
        for p in global_predictions:
            file.write(str(p))
            file.write("\n")
        acc = true_positives / total
        file.write("Accuracy: ")
        file.write(str(acc))
        file.write("MRR: ")
        file.write(str(global_mrr))


        print("Current Accuracy: %f" %(true_positives/total))

    print("Final Accuracy: %f" %(true_positives/total))


#if __name__ == "__main__":

    #Load data as a Torch Dataset object
    #dataset = DialogueHomogeneousDataset("dialogue")

    #nongraph_entity_link(dataset)