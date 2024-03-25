import os
os.environ['DGLBACKEND'] = 'pytorch'

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import time
import tqdm
import numpy as np

import dgl
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import SAGEConv
from dgl.data import DGLDataset

import sklearn.metrics
from scipy import sparse
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

# Seed for reproducible numbers
torch.manual_seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.device(device)

def scale_features(f):
    scaler = StandardScaler()
    scaler.fit(f)
    trans_data = scaler.transform(f)
    return trans_data

# use pca nor raw feature
# use_pca = True
use_pca = False

root_path = "/home/zihend1/DiseasePrediction/Data/Preprocessed/Transductive/"

samples = []

for p in os.listdir(root_path):
    if "Split" in p:
        samples.append(p)

if use_pca:
    features = np.load(root_path+"features.npy")
    features = torch.tensor(scale_features(features))
else:
    features = sparse.load_npz(root_path+"features_raw.npz")
    features = torch.tensor(scale_features(np.array(features.todense())))
    
cell_types = np.load(root_path+"cell_type.npy", allow_pickle=True)
cell_types_list = list(set(cell_types))
cell_types_list.sort()
cell_type_ids = []
for c in cell_types:
    cell_type_ids.append(cell_types_list.index(c))
cell_type_ids = torch.tensor(cell_type_ids).reshape(-1,1)

scxGNN_features = torch.hstack([cell_type_ids, features])

class ADDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="AD")

    def process(self):
        
        labels = np.load(root_path+"labels.npy")
        labels = torch.tensor(labels)
        
        edges = np.load(root_path+"edges.npy")
        edges = torch.LongTensor(edges)

        self.graph = dgl.graph(
            (edges[0,:],  edges[1,:]), num_nodes=features.shape[0]
        )
        self.graph = dgl.add_reverse_edges(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

#         self.graph.ndata["feat"] = features
        self.graph.ndata["feat"] = scxGNN_features
        self.graph.ndata["label"] = labels

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    
dataset = ADDataset()
graph = dataset[0]

print(graph)

node_features = graph.ndata["feat"]
node_labels = graph.ndata["label"]
num_features = node_features.shape[1]-1
num_classes = (node_labels.max() + 1).item()

num_cell_types = 24

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        # Create an embedding layer for cell type IDs
        self.cell_type_embedding = nn.Embedding(num_cell_types, in_feats)

        self.h_feats = h_feats
        self.dropout = nn.Dropout(0.5)

    def forward(self, mfgs, x):
        cell_type_ids = torch.tensor(x[:,0], dtype=torch.int)
        x = x[:,1:]
        # Look up the corresponding learnable weights for the cell type IDs
        cell_type_weights = self.cell_type_embedding(cell_type_ids)
        x = x * torch.sigmoid(cell_type_weights)
        
        x = self.dropout(x)
        h_dst = x[: mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---

        h = F.relu(h)        
        h = self.dropout(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---

        return h

for sample in samples:
    
#     if "AD" in sample:
#         continue
    if '7' not in sample and '8' not in sample and '9' not in sample:
        continue
    
    print(sample)
    
    data_path = "/home/zihend1/DiseasePrediction/Data/Preprocessed/Transductive/"+sample+"/"
    save_path = "/home/zihend1/DiseasePrediction/Data/Model/Transductive/"
    pred_path = "/home/zihend1/DiseasePrediction/Data/Predict/Transductive/"
    
    save_path += sample
    pred_path += sample
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    
    if use_pca:
        save_path += "/pca/"
        pred_path += "/pca/"
    else:
        save_path += "/raw/"
        pred_path += "/raw/"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
        
#     if os.path.exists(pred_path+"DGL_GraphSAGE_predict.npy"):
#         continue
        
    train_nids = np.load(data_path+"idx_train.npy")
    valid_nids = np.load(data_path+"idx_val.npy")
    test_nids = np.load(data_path+"idx_test.npy")
    
    sampler = dgl.dataloading.NeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=256,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    
    model = Model(num_features, 128, num_classes).to(device)
    
    opt = torch.optim.Adam(model.parameters())
    
    valid_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    total_val_acc = []
    best_val_acc = 0
    bad_counter = 0
    best_epoch = 0
    patience = 30
    t_total = time.time()
    
    for epoch in range(10000):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels) + 1e-4*torch.norm(torch.sigmoid(model.cell_type_embedding.weight), p=1)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )

                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )

        model.eval()

        predictions = []
        labels = []
        with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata["feat"]
                labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
            total_val_acc.append(accuracy)

        if total_val_acc[-1] > best_val_acc:
            best_val_acc = total_val_acc[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break
            
        torch.save(model.state_dict(), save_path+'DGL_scxGNN_{}.pt'.format(epoch))
        files = glob.glob(save_path+'DGL_scxGNN_*.pt')
            
        for file in files:
            epoch_nb = int(file.split('_')[-1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        torch.cuda.empty_cache()

    files = glob.glob(save_path+'DGL_scxGNN_*.pt')
        
    for file in files:
        epoch_nb = int(file.split('_')[-1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(best_epoch))
    
    model.load_state_dict(torch.load(save_path+'DGL_scxGNN_{}.pt'.format(best_epoch)))
    
    model.eval()
    
    test_dataloader = dgl.dataloading.DataLoader(
        graph,
        test_nids,
        sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    predictions = []
    labels = []
    with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata["feat"]
            labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print("Test Accuracy {}".format(accuracy))
    
    np.save(pred_path+"DGL_scxGNN_predict.npy", predictions)
        
    torch.cuda.empty_cache()

