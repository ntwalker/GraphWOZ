# GraphWOZ

This repository contains code for:
1) Transforming raw JSON formatted GraphWOZ data into a PyTorch Geometric graph Dataset with BERT encodings
2) Entity-linking by cosine similarity
3) Training and evaluating a GNN for entity-linking with GraphWOZ Data.

#Preprocessing
homogeneous_graph_transform.py transforms the raw json data to PyTorch Geometric Data objects using BERT to encode strings as node and edge attributes.
Please note: 
1) This is slow, because it does not currently use CUDA 
2) Each dialogue turn is saved as a separate graph in the "dialogues" folder
3) Only mention nodes from the current turn are encoded in the graph, and no agent responses are included (all previous turn utterances are, however, included)

#Entity-Linking
Two main files handle this: cosine_compare_entities.py contains the code for computing Precision@1 and MRR metrics for entity mention-entity links in the graph.
The method for calculating this will also output a file for the dataset containing triples in the form of (Mention, Predicted Entity, True Entity)

The file gnn_link_prediction.py imports from the previous file to first calculate these over the (train, validation, test) splits.
Subsequently, the GNN then trains on the train dataset and outputs a file with predicted links and precision@1 for the validation set.
The final step is evaluation on the test set, which outputs a similar file as before, but also including MRR calculated over the output.
