import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import Dataset

import tqdm
import os.path as osp
from os import listdir
from os.path import isfile, join

from cosine_compare_entities import nongraph_entity_link

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
        data.x = torch.reshape(data.x, (data.x.size()[0], 70*768*7))
        return data

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index, mention_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        F.dropout(x, p=0.3, training=self.training)

        z = F.softmax((x[mention_index] * x).sum(dim=1))

        return z

torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DialogueHomogeneousDataset("dialogue").shuffle()
    
train_dataset = dataset[0:int(len(dataset)/2)]
val_dataset = dataset[int(len(dataset)/2):int(len(dataset)*.75)]
test_dataset = dataset[int(len(dataset)*.75):]

#print(train_dataset.num_features)

#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = GNN(hidden_channels=64, out_channels=7)#.to(device)
#model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)

# Non-graph entity linking
#print("Calculating cosine similarities for training set...")
#nongraph_entity_link(train_dataset, "train")
#print("Calculating cosine similarities for validation set...")
#nongraph_entity_link(val_dataset, "validation")
#nongraph_entity_link(test_dataset, "test")
#print("Calculating cosine similarities for test set...")

for epoch in range(1):
    continue
    print("Training Epoch %d..." %(epoch+1))
    counter = 0
    total_loss = 0
    total_val_loss = 0
    model.train()
    for data in tqdm.tqdm(train_dataset):
        #data.to(device)
        #data.x = torch.reshape(data.x, (data.x.size()[0], 70*768*7)) # Reshape for GNN

        counter += 1
        #if counter > 20:
        #    break

        # Construct dict mapping mention node index to edge existence with the graph
        src_to_targets = dict()
        for mention_edge in data.mention_edges:
            src_index = mention_edge[0]
            if not src_index in src_to_targets: #Check if this mention exists, if not create new target array
                src_to_targets[src_index] = torch.zeros(data.x.size()[0])
            tgt_index = mention_edge[1]
            src_to_targets[src_index][tgt_index] = 1

        #for src_index, target in src_to_targets.items(): # Multiple possible targets per mention
        for mention_edge in data.mention_edges: # This way assumes one to one linking from mentions
            src_index = mention_edge[0] # Index of entity mention node (source)
            tgt_index = mention_edge[1] # Index of entity node (target)
            target = torch.zeros(data.x.size()[0], 1)#.to(device)
            target[tgt_index] = 1
            #####

            # Weight positive links higher if desired
            #weight = torch.zeros(target.size())
            #weight[data.mention_edges[0][1]] = .5
            #weight[50] = .5
            #weight = torch.flatten(weight)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, src_index)

            #criterion = torch.nn.BCELoss(weight=weight)
            criterion = torch.nn.BCELoss()
            loss = criterion(out, torch.flatten(target))
            total_loss += float(loss)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    correct = 0
    total = 0
    for data in tqdm.tqdm(val_dataset):
        target = torch.zeros(data.x.size()[0], 1)#.to(device)
        target[data.mention_edges[0][1]] = 1
        
        # Construct dict mapping mention node index to edge existence with the graph
        src_to_targets = dict()
        for mention_edge in data.mention_edges:
            src_index = mention_edge[0]
            if not src_index in src_to_targets: #Check if this mention exists, if not create new target array
                src_to_targets[src_index] = torch.zeros(data.x.size()[0])
            tgt_index = mention_edge[1]
            src_to_targets[src_index][tgt_index] = 1

        # Weight positive links higher if desired
        #weight = torch.zeros(target.size())
        #weight[data.mention_edges[0][1]] = .5
        #weight[50] = .5
        #weight = torch.flatten(weight)

            out = model(data.x, data.edge_index, src_index)
            #criterion = torch.nn.BCELoss(weight=weight)
            criterion = torch.nn.BCELoss()
            val_loss = criterion(out, torch.flatten(target))
            total_val_loss += float(val_loss)

            pair = (data.x_labels[int(src_index)], data.x_labels[int(torch.argmax(out))], data.x_labels[int(tgt_index)])
            if pair[1] == pair[2]:
                correct += 1
            total += 1
            predictions.append(pair)

    with open("epoch_" + str(epoch) + "_validation_predictions.txt", "w") as file:
        file.write("Mention, Predicted Referent Entity, True Referent Entity")
        file.write("\n-------------\n")
        for p in predictions:
            file.write(str(p))
            file.write("\n")
        acc = correct / total
        file.write("Accuracy: ")
        file.write(str(acc))

    print("Total Training loss: %f" %total_loss)
    print("Total Validation loss: %f" %total_val_loss)
    append_write = "a"
    if epoch == 0:
        append_write = "w"
    with open("results.txt", append_write) as file:
        #I know these are inelegant...
        train_loss_string = "Training Loss Epoch " + str(epoch) + ": " + str(total_loss)
        val_loss_string = "Validation Loss Epoch " + str(epoch) + ": " + str(total_val_loss)
        file.write(train_loss_string)
        file.write("\n")
        file.write(val_loss_string)
        file.write("\n\n")

torch.save(model, "linkpred_gnn.pt")

#model = torch.load("linkpred_gnn.pt")

model.eval()
predictions = []
correct = 0
total = 0
mrr = 0
for data in tqdm.tqdm(test_dataset):
    # # Construct dict mapping mention node index to edge existence with the graph
    src_to_targets = dict()
    for mention_edge in data.mention_edges:
        src_index = mention_edge[0]
        if not src_index in src_to_targets: #Check if this mention exists, if not create new target array
            src_to_targets[src_index] = torch.zeros(data.x.size()[0])
        tgt_index = mention_edge[1]
        src_to_targets[src_index][tgt_index] = 1

        out = model(data.x, data.edge_index, src_index)

        pair = (data.x_labels[int(src_index)], data.x_labels[int(torch.argmax(out))], data.x_labels[int(tgt_index)])
        
        # Calculate MRR using sorted list of indices
        for i, m in enumerate(torch.topk(out, out.size()[0])[1]):
            if m == tgt_index:
                mrr += (1/(i+1)) # Mean reciprocal rank
        if pair[1] == pair[2]:
            correct += 1
        total += 1
        predictions.append(pair)

mrr = (1/total) * mrr # Final MRR score, normalize

with open("test_results.txt", "w") as file:
    #I know these are inelegant...
    for p in predictions:
        file.write(str(p))
        file.write("\n")
    file.write("Precision@1: ")
    file.write(str(correct/total))
    file.write("\nMRR: ")
    file.write(str(mrr))
    file.write("\n\n")