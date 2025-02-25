import dgl
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Amazon, Yelp
import torch.nn.functional as F
from dgl.data.utils import load_graphs, save_graphs

def pretrain_dataloader(input_dim:int, dataset:str):
    
    if dataset == 'Yelp_Fraud' or dataset == 'Amazon_Fraud':
        if dataset == 'Yelp_Fraud':
            data = dgl.data.FraudDataset('yelp')
            dataname = 'Yelp_Fraud'
        else:
            data = dgl.data.FraudDataset('amazon')
            dataname = 'Amazon_Fraud'
        num_classes = 2
        g = data[0]
        g = dgl.to_homogeneous(g, ndata=['feature','label','train_mask','val_mask','test_mask'])
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = g.ndata['feature']
        y = g.ndata['label']

        data = Data(x=x, edge_index=edge_index, y=y)

    elif dataset == 'Amazon_Photo' or dataset == 'Amazon_Computer':
        if dataset == 'Amazon_Photo':
            data = Amazon(root='../autodl-tmp/data/', name='photo')
            dataname = 'Photo'
        elif dataset == 'Amazon_Computer':
            data = Amazon(root='../autodl-tmp/data/', name='computers')
            dataname = 'Computers'

        num_classes = data.num_classes
        data = data.data

    elif dataset == 'T-Finance':
        dataname = 'T-Finance'
        num_classes = 2
        g, label_dict = load_graphs('data/tfinance')
        g = g[0]
        g.ndata['label'] = g.ndata['label'].argmax(1)

        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = g.ndata['feature']
        y = g.ndata['label']

        data = Data(x=x, edge_index=edge_index, y=y)
        
    if data.x.shape[1] < input_dim:
        padding_size = input_dim - data.x.shape[1]
        data.x = F.pad(data.x, (0, padding_size), 'constant', 0)
    else:
        svd_reduction = SVDFeatureReduction(out_channels=input_dim) # use SVD to uniform features dimension
        data = svd_reduction(data)

    x, y = data.x, data.y
    edge_index = data.edge_index 

    x_mean = torch.mean(x, dim=0)
    x_std = torch.std(x, dim=0)
    normalized_x = (x - x_mean) / (x_std + 1e-5)

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes = len(x))
    g.ndata['feat'] = normalized_x
    g.ndata['label'] = y
    g = dgl.add_self_loop(g)

    return g, dataname, num_classes
