import time
import json
import random
import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import Dataset

from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
import tqdm
import os.path as osp
from os import listdir
from os.path import isfile, join

from cosine_compare_entities import nongraph_entity_link
from graph2text import relation_to_text, retrieve_graph_entities

#Dataset object, using preprocessed dialogue graph files stored in folder
class DialogueHomogeneousDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        datafiles = [f for f in listdir("dialogue_graphs") if isfile(join("dialogue_graphs", f))]
        idx = 0
        self.datafile_idxs = dict()
        for file in datafiles:
            self.datafile_idxs[idx] = file
            idx += 1
        return datafiles

    #def process(self):
    #    torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    #    torch.save(self.collate(self.data_list), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        data = torch.load(osp.join("dialogue_graphs", self.datafile_idxs[idx]))
        #data.x[:,1] = data.x[:,1] * data.x_mask[:,1]
        #data.x = torch.reshape(data.x, (data.x.size()[0], 70*768*7))
        return data

class NonGraphNeuralModel(torch.nn.Module):
    '''
    This model is intended to test a BERT model's ability to predict entity links without graph information
    It fine tunes BERT on the utterances up to the current dialogue turn, and predicts the entity to link to.
    Note: The entity representation, while from BERT, is NOT fine tuned. Only the predictions are.
    '''
    def __init__(self, bert):
        super().__init__()
        self._bert = BertModel.from_pretrained(bert)

        #for param in self._bert.parameters():
        #   param.requires_grad = False

        ## selective fine-tuning
        for name, child in self._bert.named_children():
            if name =='embeddings':
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

        ## FFNN
        self._fc1 = torch.nn.Linear(768, 70)
        self._dropout = torch.nn.Dropout(p=0.3)

        self.graph_head = torch.nn.Linear(768*7, 256) # 70 or 256, depending
        
    def forward(self, utterance, mask, x):
        b = self._bert(utterance)
        pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)

        x = torch.reshape(x, (x.size()[0], 70, 768*7)) #Reshape it to combine BERT and type layers

        x2 = self.graph_head(x).relu() # Process the graph's node attributes with a linear layer too

        ## FFNN
        output = self._fc1(pooler)
        f = self._dropout(output)
        b = F.relu(f)

        b = b.permute(0,2,1)
        b = b.reshape(b.size()[0], 70*256) # 70 or 256, depending

        x2 = x2.reshape(x2.size()[0], 70*256) # 70 or 256, depending

        z = F.softmax((b * x2).sum(dim=1)) # Multiply the utterance by the graph matrix and softmax to select matches
        #print(z.size())
        #z = F.softmax(z.sum(dim=1))
        
        return z

class NonGraphNeuralModelRevised(torch.nn.Module):
    '''
    This model is intended to test a BERT model's ability to predict entity links without graph information
    It fine tunes BERT on the utterances up to the current dialogue turn, and predicts the entity to link to.
    Note: The entity representation, while from BERT, is NOT fine tuned. Only the predictions are.
    '''
    def __init__(self, bert):
        super().__init__()
        self._bert = BertModel.from_pretrained(bert)
        self._bert2 = BertModel.from_pretrained(bert)

        #for param in self._bert.parameters():
        #   param.requires_grad = False
        #for param in self._bert2.parameters():
        #   param.requires_grad = False

        ## FFNN
        self._fc1 = torch.nn.Linear(768, 128)
        self._fc2 = torch.nn.Linear(768, 128)
        self._dropout = torch.nn.Dropout(p=0.3)

        self._fc11 = torch.nn.Linear(128, 32)
        self._fc22 = torch.nn.Linear(128, 32)

        self._conv1 = torch.nn.Conv1d(512, 100, 2)
        self._conv2 = torch.nn.Conv1d(1, 100, 2)
        #self._combined = torch.nn.Linear(256, 128)
        self._head = torch.nn.Linear(32, 1)
        
    def forward(self, document, mask, entities, entity_mask):
        b = self._bert(document)
        pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)

        b_entities = self._bert2(entities)
        #pooler_entities = b.last_hidden_state[:, entity_mask].diagonal().permute(2, 0, 1)


        ## FFNN on document
        output = self._fc1(pooler.mean(dim=1))
        f = self._dropout(output)
        #b = F.tanh(f)
        #b = self._fc11(b)

        ## FFNN on entities
        output2 = self._fc2(b_entities.last_hidden_state.mean(dim=1))
        f2 = self._dropout(output2)
        #b2 = F.tanh(f2)
        #b2 = self._fc22(b2)

        #b = b.reshape(b.size()[0], 512*128)
        #b2 = b2.reshape(b2.size()[0], 10*128)

        #b = self._conv1(b)
        #print(b.size())
        #b2 = self._conv2(b2)

        #b = b.mean(dim=1) # Average over the BERT embedding dimension
        #b2 = b2.mean(dim=1)

        #print("b Sizes")
        #print(f.size())
        #print(f2.size())

        #z = (b * b2).sum(dim=1).sigmoid() # "Works"

        z = (f * f2).sum(dim=1).sigmoid()
        #print(z.sum(dim=1).size())
        #z = self._combined(z)
        #z = self._head(z).sigmoid().squeeze()
        #print(z)
        #print(z.size())
        #time.sleep(299)
        
        return z

# Text representations of graphs
with open("graphwoz_04072022.json", "r") as file:
  json_data = json.load(file)

docs = relation_to_text(json_data)
entities, mentions, mention_strings = retrieve_graph_entities(json_data)

#temp = list(zip(docs, entities, mentions))
#random.shuffle(temp)
#docs, entities, mentions = zip(*temp)

torch.manual_seed(1234)
random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DialogueHomogeneousDataset("dialogue").shuffle()

##I should have saved the dialogue/turn IDs with the graphs but didn't, so this is a workaround to make sure the train/dev/test split is the same between GNN and non-GNN
print("Aligning dataset...")
new_docs = []
new_ents = []
new_ments = []
new_strings = []
old_set = set(docs)
for d in dataset:
    mns = [d.x_labels[m[0]] for m in d.mention_edges]
    for i in range(len(docs)):
        if set(mns) == set(mention_strings[i]):
        #if docs[i][3] == d.x_labels[-(len(d.mention_edges)+1)]: ## Not quite correct
            new_docs.append(docs[i])
            new_ents.append(entities[i])
            new_ments.append(mentions[i])
            new_strings.append(mention_strings[i])
            #print("Match")
            break

print(len(new_docs))
#print(old_set.difference(set(new_docs)))
##Ugly
docs = new_docs
entities = new_ents
mentions = new_ments
mention_strings = new_strings
    
train_dataset = dataset[0:int(len(dataset)/2)]
val_dataset = dataset[int(len(dataset)/2):int(len(dataset)*.75)]
test_dataset = dataset[int(len(dataset)*.75):]

train_docs = docs[0:int(len(docs)/2)]
val_docs = docs[int(len(docs)/2):int(len(docs)*.75)]
test_docs = docs[int(len(docs)*.75):]
train_ents = entities[0:int(len(docs)/2)]
val_ents = entities[int(len(docs)/2):int(len(docs)*.75)]
test_ents = entities[int(len(docs)*.75):]
train_mentions = mentions[0:int(len(docs)/2)]
val_mentions = mentions[int(len(docs)/2):int(len(docs)*.75)]
test_mentions = mentions[int(len(docs)*.75):]
train_strings = mention_strings[0:int(len(docs)/2)]
val_strings = mention_strings[int(len(docs)/2):int(len(docs)*.75)]
test_strings = mention_strings[int(len(docs)*.75):]


#for i, mentions in enumerate(train_mentions):
#    for m in mentions:
#        print(train_ents[i][m])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#bert = BertModel.from_pretrained("bert-base-cased")
bertpath = "bert-base-cased"

#model = NonGraphNeuralModel(bertpath)
model = NonGraphNeuralModelRevised(bertpath).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(40):
    model.train()
    total_loss = 0
    total_val_loss = 0
    for dialogue_index, dialogue_mention_indices in enumerate(tqdm.tqdm(train_mentions)):
        if dialogue_mention_indices != []:
            entity_tokens = tokenizer(train_ents[dialogue_index], return_tensors='pt', max_length=10, padding="max_length", truncation=True)
            dialogue_mentions = [train_ents[dialogue_index][ment] for ment in train_mentions[dialogue_index]]
            mention_tokens = tokenizer(dialogue_mentions, return_tensors='pt', max_length=10, padding="max_length", truncation=True)
            doc_tokens = tokenizer(train_docs[dialogue_index][2], return_tensors='pt', max_length=512, padding="max_length", truncation=True)

            target = torch.zeros(entity_tokens["input_ids"].size()[0], 1).to(device)

            # Weight positive links higher if desired
            weight = torch.zeros(target.size())
            selection = random.choice([i for i in range(0,target.size()[0]) if i in dialogue_mention_indices])
            weight[selection] = .5
            for i in range(len(weight)):
                if weight[i] != .5:
                    weight[i] = .5 / (len(weight))
            #selection = random.choice([i for i in range(0,target.size()[0]) if i not in dialogue_mention_indices])
            #weight[selection] = 0
            weight = torch.flatten(weight).to(device)
                
            for ment_index in dialogue_mention_indices: # Add 1 for each target entity to link!
                target[ment_index] = 1
            #target[dialogue_mention_indices[0]] = 1 # Only target the first mention

            optimizer.zero_grad()
            out = model(doc_tokens["input_ids"].to(device), doc_tokens["attention_mask"].to(device), entity_tokens["input_ids"].to(device), entity_tokens["attention_mask"].to(device))

            #if epoch > 5:
            #    print(out)
            criterion = torch.nn.BCELoss()
            #criterion = torch.nn.BCELoss(weight=weight)
            loss = criterion(out, torch.flatten(target))
            total_loss += float(loss)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    correct = 0
    total = 0
    mrr = 0
    for dialogue_index, dialogue_mention_indices in enumerate(tqdm.tqdm(val_mentions)):
        if dialogue_mention_indices != []:
            entity_tokens = tokenizer(val_ents[dialogue_index], return_tensors='pt', max_length=10, padding="max_length", truncation=True)
            dialogue_mentions = [val_ents[dialogue_index][ment] for ment in val_mentions[dialogue_index]]
            mention_tokens = tokenizer(dialogue_mentions, return_tensors='pt', max_length=10, padding="max_length", truncation=True)
            doc_tokens = tokenizer(val_docs[dialogue_index][2], return_tensors='pt', max_length=512, padding="max_length", truncation=True)

            target = torch.zeros(entity_tokens["input_ids"].size()[0], 1).to(device)

            # Weight positive links higher if desired
            weight = torch.zeros(target.size())
            selection = random.choice([i for i in range(0,target.size()[0]) if i in dialogue_mention_indices])
            weight[selection] = .5
            for i in range(len(weight)):
                if weight[i] != .5:
                    weight[i] = .5 / (len(weight))
            #selection = random.choice([i for i in range(0,target.size()[0]) if i not in dialogue_mention_indices])
            #weight[selection] = 0
            weight = torch.flatten(weight).to(device)
                
            for ment_index in dialogue_mention_indices: # Add 1 for each target entity to link!
                target[ment_index] = 1
            #target[dialogue_mention_indices[0]] = 1 # Only target the first mention

            optimizer.zero_grad()
            out = model(doc_tokens["input_ids"].to(device), doc_tokens["attention_mask"].to(device), entity_tokens["input_ids"].to(device), entity_tokens["attention_mask"].to(device))

            #print(out)
            criterion = torch.nn.BCELoss()
            #criterion = torch.nn.BCELoss(weight=weight)
            loss = criterion(out, torch.flatten(target))
            total_val_loss += float(loss)

            #pair = (val_ents[dialogue_index][int(dialogue_mention_indices[0])], val_ents[dialogue_index][int(torch.argmax(out))], val_ents[dialogue_index][int(data.mention_edges[0])], float(out[torch.argmax(out)]))
            ## Single mention node to single referent
            if len(val_mentions[dialogue_index]) == 1:
                pair = (val_strings[dialogue_index][0], val_ents[dialogue_index][int(torch.argmax(out))], val_ents[dialogue_index][int(dialogue_mention_indices[0])], float(out[torch.argmax(out)]))
                if pair[1] == pair[2]:
                    correct += 1
                total += 1
            ## Multiple mentions
            else:
                pred_ents = [val_ents[dialogue_index][int(ent)] for ent in torch.topk(out, len(val_strings[dialogue_index]))[1]]
                true_ents = [ent for i, ent in enumerate(val_ents[dialogue_index]) if i in dialogue_mention_indices]
                probs = [float(prob) for prob in torch.topk(out, len(val_strings[dialogue_index]))[0]]
                pair = (val_strings[dialogue_index], pred_ents, true_ents, probs)
                for i in range(len(pair[1])):
                    if pair[1][i] == pair[2][i]:
                        correct += 1
                    total += 1

            predictions.append(pair)

    if (epoch+1) % 5 == 0:
        with open("epoch_" + str(epoch+1) + "_nongraph_validation_predictions.txt", "w") as file:
            file.write("Mention, Predicted Referent Entity, True Referent Entity")
            file.write("\n-------------\n")
            for p in predictions:
                file.write(str(p))
                file.write("\n")
            acc = correct / total
            file.write("Precision: ")
            file.write(str(acc))

    print("Training Loss at Epoch %d: %f" %((epoch + 1), total_loss))
    print("Validation Loss at Epoch %d: %f" %((epoch + 1), total_val_loss))

    append_write = "a"
    if epoch == 0:
        append_write = "w"
    with open("nongraph_epoch_losses.txt", append_write) as file:
        #I know these are inelegant...
        train_loss_string = "Training Loss Epoch " + str(epoch) + ": " + str(total_loss)
        val_loss_string = "Validation Loss Epoch " + str(epoch) + ": " + str(total_val_loss)
        file.write(train_loss_string)
        file.write("\n")
        file.write(val_loss_string)
        file.write("\n\n")

#torch.save(model, "nongraph_linkpred.pt")

model = torch.load("nongraph_linkpred.pt")

model.eval()
predictions = []
correct = 0
total = 0
mrr = 0
for dialogue_index, dialogue_mention_indices in enumerate(tqdm.tqdm(test_mentions)):
    if dialogue_mention_indices != []:
        entity_tokens = tokenizer(test_ents[dialogue_index], return_tensors='pt', max_length=10, padding="max_length", truncation=True)
        dialogue_mentions = [test_ents[dialogue_index][ment] for ment in test_mentions[dialogue_index]]
        mention_tokens = tokenizer(dialogue_mentions, return_tensors='pt', max_length=10, padding="max_length", truncation=True)
        doc_tokens = tokenizer(test_docs[dialogue_index][2], return_tensors='pt', max_length=512, padding="max_length", truncation=True)

        target = torch.zeros(entity_tokens["input_ids"].size()[0], 1).to(device)
            
        for ment_index in dialogue_mention_indices: # Add 1 for each target entity to link!
            target[ment_index] = 1
        #target[dialogue_mention_indices[0]] = 1 # Only target the first mention

        optimizer.zero_grad()
        out = model(doc_tokens["input_ids"].to(device), doc_tokens["attention_mask"].to(device), entity_tokens["input_ids"].to(device), entity_tokens["attention_mask"].to(device))

        #pair = (test_ents[dialogue_index][int(dialogue_mention_indices[0])], test_ents[dialogue_index][int(torch.argmax(out))], val_ents[dialogue_index][int(data.mention_edges[0])], float(out[torch.argmax(out)]))
        pair = (test_strings[dialogue_index], test_ents[dialogue_index][int(torch.argmax(out))], test_ents[dialogue_index][int(dialogue_mention_indices[0])], float(out[torch.argmax(out)]))
        if pair[1] == pair[2]:
            correct += 1
        # Calculate MRR using sorted list of indices
        for i, m in enumerate(torch.topk(out, out.size()[0])[1]):
            if m in dialogue_mention_indices:
                mrr += (1/(i+1)) # Mean reciprocal rank
        total += 1
        predictions.append(pair)

mrr = (1/total) * mrr # Final MRR score, normalize

with open("test_nongraph_validation_predictions.txt", "w") as file:
    file.write("Mention, Predicted Referent Entity, True Referent Entity")
    file.write("\n-------------\n")
    for p in predictions:
        file.write(str(p))
        file.write("\n")
    file.write("Precision@1: ")
    file.write(str(correct/total))
    file.write("\nMRR: ")
    file.write(str(mrr))
    file.write("\n\n")

## Using Graph Dialogue Dataset (Below this line)
## Utterance to BERT, then BERT to Entity link model
# for epoch in range(20):
#     model.train()
#     total_loss = 0
#     total_val_loss = 0
#     for i, data in enumerate(tqdm.tqdm(train_dataset)):
#         utterances = ""
#         for label in data.x_labels:
#             if len(label.split()) > 3:
#                 #print(label)
#                 utterances += label
#                 utterances += ". "
#         utterances = utterances.strip()
#         #print(utterances)
#         #print("\n")

#         target = torch.zeros(data.x.size()[0], 1)#.to(device)
#         target[data.mention_edges[0][1]] = 1

#         # Weight positive links higher if desired
#         weight = torch.zeros(target.size())
#         weight[data.mention_edges[0][1]] = .5
#         selection = random.choice([i for i in range(0,target.size()[0]) if i != data.mention_edges[0][1]])
#         weight[selection] = .5
#         weight = torch.flatten(weight)

#         ##Use concetenated user utterances ONLY
#         #tokens = tokenizer(utterances, return_tensors='pt', max_length=70, padding="max_length", truncation=True)
#         ##Use verbalized graphs + user utterances + system responses
#         tokens = tokenizer(train_docs[i][2], return_tensors='pt', max_length=256, padding="max_length", truncation=True)

#         #print(tokenizer.tokenize(train_docs[i][2], max_length=256))
        
#         out = model(tokens["input_ids"], tokens["attention_mask"], data.x)

#         #print((data.x_labels[int(data.mention_edges[0][0])], data.x_labels[int(torch.argmax(out))], data.x_labels[int(graph.mention_edges[0][1])]))
        
#         #print(torch.argmax(out), data.mention_edges[0][0])
        
#         #criterion = torch.nn.BCELoss()
#         criterion = torch.nn.BCELoss(weight=weight)
#         loss = criterion(out, torch.flatten(target))
#         total_loss += float(loss)
#         loss.backward()
#         optimizer.step()
    
#     model.eval()
#     predictions = []
#     correct = 0
#     total = 0
#     mrr = 0
#     for i, data in enumerate(tqdm.tqdm(val_dataset)):
#         target = torch.zeros(data.x.size()[0], 1)#.to(device)
#         target[data.mention_edges[0][1]] = 1

#         # Weight positive links higher if desired
#         weight = torch.zeros(target.size())
#         weight[data.mention_edges[0][1]] = .5
#         selection = random.choice([i for i in range(0,target.size()[0]) if i != data.mention_edges[0][1]])
#         weight[selection] = .5
#         weight = torch.flatten(weight)

#         ##Use concetenated user utterances ONLY
#         #tokens = tokenizer(utterances, return_tensors='pt', max_length=70, padding="max_length", truncation=True)
#         ##Use verbalized graphs + user utterances + system responses
#         tokens = tokenizer(val_docs[i][2], return_tensors='pt', max_length=256, padding="max_length", truncation=True)
        
#         #out = model(data.x, data.x_mask, data.edge_index, src_index)
#         out = model(tokens["input_ids"], tokens["attention_mask"], data.x)
#         #criterion = torch.nn.BCELoss()
#         criterion = torch.nn.BCELoss(weight=weight)
#         val_loss = criterion(out, torch.flatten(target))
#         total_val_loss += float(val_loss)

#         #for i, m in enumerate(torch.topk(out, out.size()[0])[1]):
#         #if m == tgt_index:
#         #    mrr += (1/(i+1)) # Mean reciprocal rank

#         pair = (data.x_labels[int(data.mention_edges[0][0])], data.x_labels[int(torch.argmax(out))], data.x_labels[int(data.mention_edges[0][1])], float(out[torch.argmax(out)]))
#         if pair[1] == pair[2]:
#             correct += 1
#         total += 1
#         predictions.append(pair)

#     with open("epoch_" + str(epoch) + "_nongraph_validation_predictions.txt", "w") as file:
#         file.write("Mention, Predicted Referent Entity, True Referent Entity")
#         file.write("\n-------------\n")
#         for p in predictions:
#             file.write(str(p))
#             file.write("\n")
#         acc = correct / total
#         file.write("Precision: ")
#         file.write(str(acc))

#     print("Training Loss at Epoch %d: %f" %((epoch + 1), total_loss))
#     print("Total Validation loss: %f" %total_val_loss)
#     # append_write = "a"
#     # if epoch == 0:
#     #     append_write = "w"
#     # with open("results.txt", append_write) as file:
#     #     #I know these are inelegant...
#     #     train_loss_string = "Training Loss Epoch " + str(epoch) + ": " + str(total_loss)
#     #     val_loss_string = "Validation Loss Epoch " + str(epoch) + ": " + str(total_val_loss)
#     #     file.write(train_loss_string)
#     #     file.write("\n")
#     #     file.write(val_loss_string)
#     #     file.write("\n\n")


# model.save("nongraph_linkpred_model.pt")