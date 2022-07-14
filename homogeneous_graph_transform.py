import torch
import json
import time
import copy
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url

class DialogueDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])

def json_to_graph_indices(jdata):
    '''
    Function to convert GraphWOZ JSON data of dialogues to indices and string representations (attributes)
    All relations are converted to a string describing the relation
    Each turn of each dialogue is a separate set of indices and attributes (a distinct graph)
    This function returns: Node indices and attributes, edge indices and attributes
    N.B. This representation of the data must be further transformed (node and edge strings must be vectorized)
    '''

    dialogue_ents = dict() # Dialogue to turn, turn to node names and types
    ent_id_to_idx = dict() # Dialogue to turn, turn to entity ids, ids to indexes
    ent_name_to_id = dict() # Dialogue to turn, turn to entity names, names to ids
    dialogue_edges = dict() # Dialogue to turn, turn to edge indices, attributes, and masks
    edge_indices = dict() # Dialogue to turn, turn to edge indices
    edge_attributes = dict() # Dialogue to turn, turn to edge attributes (strings)

    #Create the nodes in the graph
    for dialogue in jdata:
        dialogue_ents[dialogue] = dict() # Dialogue entities as strings (attributes), along with type (parallel lists)
        ent_id_to_idx[dialogue] = dict() # Dialogue entities ID to node index mapping
        ent_name_to_id[dialogue] = dict() # Dialogue entities name to id mapping
        edge_indices[dialogue] = dict() # Dialogue edges
        edge_attributes[dialogue] = dict() # Dialogue edge attributes, as strings

        dialogue_edges[dialogue] = dict() # Dialogue edge attributes, as strings

        prev_utterances = [] # Save utterances from previous turns
        prev_mentions = [] # Save mentions from previous turns
        
        for turn in jdata[dialogue]["log"]:
            
            curr_idx = 0 # Index of added entity, must be incremented each time an entity is added
            dialogue_ents[dialogue][turn["turn"]] = {"entities" : [], "types" : []}
            dialogue_edges[dialogue][turn["turn"]] = {"edge_indices" : [], "edge_attributes" : [], "mention_edges" : []}
            ent_id_to_idx[dialogue][turn["turn"]] = dict()
            ent_name_to_id[dialogue][turn["turn"]] = dict()
            
            edge_indices[dialogue][turn["turn"]] = [] # A [2, num_edges] list, to be converted to a tensor
            edge_attributes[dialogue][turn["turn"]] = [] # List of strings, parallel with the indices. I.e. attr 0 should be the text of edge 0
            
            #Base data: entities from calendar. Also create offices, times, and other attributes as new nodes
            for ent_type in jdata[dialogue]["data"]:
                if ent_type in ["events", "people"]:
                    for ent in jdata[dialogue]["data"][ent_type]:
                        dialogue_ents[dialogue][turn["turn"]]["entities"].append(ent["name"])
                        dialogue_ents[dialogue][turn["turn"]]["types"].append(ent_type)
                        ent_id_to_idx[dialogue][turn["turn"]][ent["id"]] = curr_idx
                        ent_name_to_id[dialogue][turn["turn"]][ent["name"]] = ent["id"]
                        curr_idx += 1
                        
                        #Create attributes as full nodes
                        if "office" in ent:
                            dialogue_ents[dialogue][turn["turn"]]["entities"].append(ent["office"])
                            dialogue_ents[dialogue][turn["turn"]]["types"].append("rooms")
                            ent_id_to_idx[dialogue][turn["turn"]][ent["office"]] = curr_idx
                            #print(ent["office"], curr_idx)
                            curr_idx += 1
                            ent_name_to_id[dialogue][turn["turn"]][ent["office"]] = ent["office"] # Name = ID for offices
                else:
                    for ent in jdata[dialogue]["data"][ent_type]:
                        dialogue_ents[dialogue][turn["turn"]]["entities"].append(ent)
                        dialogue_ents[dialogue][turn["turn"]]["types"].append(ent_type)
                        ent_id_to_idx[dialogue][turn["turn"]][jdata[dialogue]["data"][ent_type][ent]["id"]] = curr_idx
                        ent_name_to_id[dialogue][turn["turn"]][ent] = jdata[dialogue]["data"][ent_type][ent]["id"]
                        curr_idx += 1

            #Connect calendar data with edges. Each edge attribute is a string describing the relation e.g. "Meeting is held in the Beta conference room"
            for ent_type in jdata[dialogue]["data"]:
                if ent_type in ["events", "people"]:
                    for ent in jdata[dialogue]["data"][ent_type]:
                        for attr in ent:
                            if attr == "attendees":
                                for person in ent[attr]:
                                    source_id = ent_name_to_id[dialogue][turn["turn"]][person]
                                    source_idx = ent_id_to_idx[dialogue][turn["turn"]][source_id]
                                    target_idx = ent_id_to_idx[dialogue][turn["turn"]][ent["id"]]
                                    
                                    #Add edge index and attribute
                                    edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([source_idx, target_idx])))
                                    edge_attributes[dialogue][turn["turn"]].append(person + " is attending " + ent["name"])

                                    #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge
                                    #dialogue_edges[dialogue][turn["turn"]]["edge_indices"].append(np.transpose(np.array([source_idx, target_idx])))
                                    #dialogue_edges[dialogue][turn["turn"]]["edge_attributes"].append(person + " is attending " + ent["name"])
                            if attr == "group":
                                source_idx = ent_id_to_idx[dialogue][turn["turn"]][ent["id"]]
                                target_id = ent_name_to_id[dialogue][turn["turn"]][ent["group"]]
                                target_idx = ent_id_to_idx[dialogue][turn["turn"]][target_id]
                                
                                #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge
                                edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([source_idx, target_idx])))
                                edge_attributes[dialogue][turn["turn"]].append(ent["name"] + " is a part of the " + ent["group"] + " group")
                            if attr == "location":
                                source_idx = ent_id_to_idx[dialogue][turn["turn"]][ent["id"]]
                                target_id = ent_name_to_id[dialogue][turn["turn"]][ent["location"]]
                                target_idx = ent_id_to_idx[dialogue][turn["turn"]][target_id]
                                
                                #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge
                                edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([source_idx, target_idx])))
                                if len(ent["location"]) == 3:
                                    edge_attributes[dialogue][turn["turn"]].append(ent["name"] + " will be held in room " + ent["location"])
                                else:
                                    edge_attributes[dialogue][turn["turn"]].append(ent["name"] + " will be held in the " + ent["location"] + " conference room")  
                            if attr == "organizer":
                                source_id = ent_name_to_id[dialogue][turn["turn"]][ent["organizer"]]
                                source_idx = ent_id_to_idx[dialogue][turn["turn"]][source_id]
                                target_idx = ent_id_to_idx[dialogue][turn["turn"]][ent["id"]]
                                
                                #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge
                                edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([source_idx, target_idx])))
                                edge_attributes[dialogue][turn["turn"]].append(ent["organizer"] + " is organizing " + ent["name"])
                                    
                            #if attr == "start_time":
                            #if attr == "end_time":
                            #else:
                            #    print(attr)                       
                        
            #Create utterances as nodes, add in all previous turn utterances
            current_utterance_text = turn["alternative"][0]["transcript"]
            last_utt_idx = -1 # This will be the last of the utterances added, used to link with the current mentions
            prev_utt_idx = -1 # Idx of last utterance if it exists, to link with current utterance

            prev_utterances.append(current_utterance_text)

            for i, utterance in enumerate(prev_utterances): # Add all utterances up to previous turn as entities

                dialogue_ents[dialogue][turn["turn"]]["entities"].append(utterance)
                dialogue_ents[dialogue][turn["turn"]]["types"].append("utterance")
                ent_id_to_idx[dialogue][turn["turn"]][utterance] = curr_idx #ID is the text because it's easy
                utterance_idx = ent_id_to_idx[dialogue][turn["turn"]][utterance] #Save utterance index to link to mentions
                ent_name_to_id[dialogue][turn["turn"]][utterance] = utterance #Same as ID

                if prev_utt_idx != -1:
                    # Create edge from previous utterance to current utterance
                    #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge
                    edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([prev_utt_idx, curr_idx])))
                    edge_attributes[dialogue][turn["turn"]].append(prev_utterances[i-1] + " is followed by " + utterance)

                prev_utt_idx = copy.deepcopy(curr_idx) # Update index of previous utterance at turn T
                last_utt_idx = copy.deepcopy(curr_idx) # Update index of final utterance at turn T
                curr_idx += 1

            #for i, utterance in enumerate(prev_mentions):

            
            #Entity mentions from utterances, link from utterance to mentions, and links from mentions to calendar entities
            last_mention = ""
            for element in turn["asr_entities"].split(", "):
                s = element.split(" : ")
                if len(s) == 2 and not "@" in s[1]:
                    
                    #Append entity mention as a node
                    if s[0] == last_mention:
                        source_node_idx = ent_id_to_idx[dialogue][turn["turn"]][s[0]]
                    else:
                        dialogue_ents[dialogue][turn["turn"]]["entities"].append(s[0])
                        dialogue_ents[dialogue][turn["turn"]]["types"].append("mention")
                        ent_id_to_idx[dialogue][turn["turn"]][s[0]] = curr_idx
                        source_node_idx = ent_id_to_idx[dialogue][turn["turn"]][s[0]]
                        curr_idx += 1
                    
                    
                    #Target node of mentions
                    try:
                        ent_id = ent_name_to_id[dialogue][turn["turn"]][s[1]]
                    except:
                        dialogue_ents[dialogue][turn["turn"]]["entities"].append(s[1])
                        ent_id = s[1]
                        ent_id_to_idx[dialogue][turn["turn"]][ent_id] = curr_idx
                        curr_idx += 1
                        ent_name_to_id[dialogue][turn["turn"]][ent] = jdata[dialogue]["data"][ent_type][ent]["id"]
                    
                    target_node_idx = ent_id_to_idx[dialogue][turn["turn"]][ent_id]
                    
                    #Connect mention to referent node, "mention refers to entity"
                    edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([source_node_idx, target_node_idx])))
                    edge_attributes[dialogue][turn["turn"]].append(s[0] + " refers to " + s[1])
                    #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(1) #Is a mention edge
                    dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(np.transpose(np.array([source_node_idx, target_node_idx]))) #Is a mention edge, save INDEX of this edge

                    #Connect current utterance to current mentions, "utterance mentions mention"
                    edge_indices[dialogue][turn["turn"]].append(np.transpose(np.array([last_utt_idx, source_node_idx])))
                    edge_attributes[dialogue][turn["turn"]].append(prev_utterances[-1] + " mentions " + s[0])
                    #dialogue_edges[dialogue][turn["turn"]]["mention_edges"].append(0) #Not a mention edge (mention must be source)
                    
                    last_mention = s[0]
                    #if s[1] == "@speaker":     

                #Unlinked Entity mentions
                #if len(s) == 1:


            dialogue_edges[dialogue][turn["turn"]]["edge_indices"] = edge_indices[dialogue][turn["turn"]] 
            dialogue_edges[dialogue][turn["turn"]]["edge_attributes"] = edge_attributes[dialogue][turn["turn"]] 


    return dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx

def encode_graphs(dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx, tokenizer, bert):
    '''
    Function to transform raw string data into PyTorch Geometric graph objects
    Input: Set of node indices and attributes, edge indices and attributes, tokenizer and BERT model
    Output: Set of dialogue, turn graphs as PyTorch Geometric objects
    Each node and each edge string is encoded with BERT. These are the node and edge attributes in the graph.
    ----
    To be added: Negative edges that do not exist from mentions to entities in the graph
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    bert = bert.to(device)
    
    data_list = []
    node_attr = []
    edge_attr = []

    #Hard coded for now, 6 dimensions one hot encoded
    type_to_onehot = {"people": 0, "events": 1, "rooms": 3, "groups": 4, "utterance": 5, "mention": 6}

    processed = 0
    #BERT encoding of node and edge attributes
    for dialogue, turns in dialogue_ents.items():
        for t, ent_dict in turns.items():

            print("Processing dialogue: ", dialogue, " turn: ", t)

            #Skip early dialogues for testing
            if int(dialogue.split("_")[1]) < 123:
                continue

            #Cut off processing after point
            #if int(dialogue.split("_")[1]) > 122:
            #    break

            #for i, e in enumerate(dialogue_ents[dialogue][t]["entities"]):
            #    if dialogue_ents[dialogue][t]["types"][i] == "mention":
            #        print(ent_id_to_idx[dialogue][t][e])

            mention_node_indices = [ent_id_to_idx[dialogue][t][e] for i, e in enumerate(dialogue_ents[dialogue][t]["entities"]) if dialogue_ents[dialogue][t]["types"][i] == "mention"]
            mention_edges = [e for e in dialogue_edges[dialogue][t]["edge_indices"] if e[0] in mention_node_indices]
            print(mention_edges)
            if mention_edges == []:
                continue

            #Encode node attributes, still memory overload
            #ent_tensors= torch.Tensor()#.to(device)
            #for ent in ent_dict["entities"]:
            #    ent_tokens = tokenizer([ent], return_tensors='pt', max_length=60, padding="max_length")#['input_ids']
            #    ent_encoded = bert(ent_tokens['input_ids'].to(device)).last_hidden_state.to('cpu') #Process on CUDA, save OFF CUDA
            #    ent_tensors = torch.cat((ent_tensors, ent_encoded), 0)

            # Original, runs into memory overload
            ent_tokens = tokenizer(ent_dict["entities"], return_tensors='pt', max_length=70, padding="max_length")#['input_ids']
            ent_tensors = bert(ent_tokens['input_ids'].to(device)).last_hidden_state

            # Create type encoding as one hot vectors, to be a new dimension
            onehot_types = F.one_hot(torch.Tensor([type_to_onehot[key] for key in dialogue_ents[dialogue][t]["types"]]).long())
            
            #Annoying reshaping of the entities tensor
            ent_tensors = ent_tensors.unsqueeze(3) # Add new dimension
            ent_tensors = ent_tensors.expand(int(ent_tensors.size()[0]), 70, 768, int(onehot_types.size()[1])) # Expand new dimension
            ent_tensors = ent_tensors[:,:,:,onehot_types[1]] # Insert one hot type encoding in new dimension

            #BERT Encoding of strings as edge attributes

            #Still memory overload this way
            #edge_tensors= torch.Tensor()#.to(device)
            #edges = dialogue_edges[dialogue][t]["edge_attributes"]
            #for e in edges:
            #    edge_tokens = tokenizer([e], return_tensors='pt', max_length=60, padding="max_length")
            #    print(edge_tokens.size())
            #    edge_encoded = bert(edge_tokens['input_ids'].to(device)).last_hidden_state.to('cpu')
            #    edge_tensors = torch.cat((edge_tensors, edge_encoded), 0)


            # Original, runs into memory overload
            edges = dialogue_edges[dialogue][t]["edge_attributes"]
            edge_tokens = tokenizer(edges, return_tensors='pt', max_length=70, padding="max_length")
            edge_tensors = bert(edge_tokens['input_ids'].to(device)).last_hidden_state

            #node_attr.append(ent_tensors)
            #edge_attr.append(edge_tensors)

            g_edges = np.array(dialogue_edges[dialogue][t]["edge_indices"])
            g_edges = torch.Tensor(g_edges).T.long() # Indices must be [2, n] edges, long tensor

            m_edges = np.array(dialogue_edges[dialogue][t]["mention_edges"])

            #Create negative example edges
            #neg_m_edges = 

            #Put encoded data into PyTorch Geometric object, target y is all 1 because all edges exist
            data = Data(x=ent_tensors, x_mask=ent_tokens["attention_mask"], x_labels=dialogue_ents[dialogue][t]["entities"], edge_index=g_edges, edge_attr=edge_tensors, mention_edges=m_edges, y=torch.ones(g_edges.size()[1]))
            
            #Save each dialogue:turn graph as a separate .pt file so as not to overload memory on CUDA?
            torch.save(data, "dialogue_graphs/dialogue_graph_" + str(processed + 1) + "_" + str(t) + ".pt")

            #data_list.append(data)

        processed += 1
        print("Processed %d dialogues" %processed)

        #Save data list to disk in batches of 5 dialogues (may still be too much memory)
        #if processed % 5 == 0:
        #    torch.save(data_list, "dialogue_graphs_" + str(processed) + ".pt") # Save in batches of 5
        #    print("Saved batch")
        #    data_list = [] # Clear list to free memory

        # Cap the number of dialogues to process if needed, preprocessing is slow
        #if processed == 10:
        #    print("DONE")
        #    break
    print("DONE")


    return data_list, node_attr, edge_attr

if __name__ == "__main__":
    
    with open("graphwoz_04072022.json", "r") as file:
        data = json.load(file)

    dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx = json_to_graph_indices(data)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    bert = BertModel.from_pretrained("bert-base-cased")

    data_list, node_attr, edge_attr = encode_graphs(dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx, tokenizer, bert)
    #Old version
    #data_list, node_attr, edge_attr = encode_graphs(dialogue_ents, ent_name_to_id, ent_id_to_idx, edge_indices, edge_attributes, tokenizer, bert)