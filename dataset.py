from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import os.path as osp
import random
from dgl import heterograph
from paths import PRETRAINED_ROOT, cached_dataset, raw_dataset

class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs(str(raw_dataset('tfinance')))
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs(str(raw_dataset('tfinance')))
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs(str(raw_dataset('tfinance')))
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                new_id = random.sample(normal_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = label

        elif name == 'tsocial':
            graph, label_dict = load_graphs(str(raw_dataset('tsocial')))
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'Twibot-20':
            path = str(raw_dataset('Twibot-20_raw'))

            if not osp.exists(path):
                raise KeyError("Dataset path not found")

            labels = torch.load(osp.join(path, 'label.pt'))
            des_embedding = torch.load(osp.join(path, 'des_tensor.pt')).contiguous()
            tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt')).contiguous()
            num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt')).contiguous()
            cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt')).contiguous()
            edge_index = torch.load(osp.join(path, 'edge_index.pt')).contiguous()
            edge_type = torch.load(osp.join(path, 'edge_type.pt')).contiguous()

            # Load the pretrained embedding as the primary feature.
            node_features = torch.load(PRETRAINED_ROOT / 'Twibot-20_pretrained_embedding_0.pt').contiguous()

            # Add nodes and edges
            num_nodes = des_embedding.shape[0]
            src, dst = edge_index
            # Separate edges by type.
            src_0, dst_0 = src[edge_type == 0], dst[edge_type == 0]
            src_1, dst_1 = src[edge_type == 1], dst[edge_type == 1]

            # Build a heterogeneous graph with '_E0' and '_E1' edge types.
            graph_data = {
                ('_N', '_E0', '_N'): (src_0, dst_0),
                ('_N', '_E1', '_N'): (src_1, dst_1)
            }
            graph = heterograph(graph_data)

            random_values = torch.randint(0, 2, (num_nodes - labels.shape[0],), dtype=torch.long)
            labels = torch.cat([labels, random_values])
            train_idx = torch.arange(0, 8278)
            val_idx = torch.arange(8278, 8278 + 2365)
            test_idx = torch.arange(8278 + 2365, 8278 + 2365 + 1183)
# Add the original features to the graph data.
            graph.ndata['label'] = labels
            graph.ndata['feature'] = node_features
            # Attach the original feature tensors as well.
            graph.ndata['des_embedding'] = des_embedding
            graph.ndata['tweet_embedding'] = tweet_embedding
            graph.ndata['num_property_embedding'] = num_property_embedding
            graph.ndata['cat_property_embedding'] = cat_property_embedding
            
            graph.ndata['train_mask'] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True)
            graph.ndata['val_mask'] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True)
            graph.ndata['test_mask'] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True)

            # Attach edge-type information to the graph edge data.
            graph.edges['_E0'].data['edge_type'] = torch.zeros(len(src_0), dtype=torch.long)
            graph.edges['_E1'].data['edge_type'] = torch.ones(len(src_1), dtype=torch.long)

            save_graphs(str(cached_dataset('Twibot-20')), [graph], {'labels': labels})
        elif name == 'Twibot-22':
            # Load pre-built DGL heterograph (already has labels, masks, features)
            graph_path = raw_dataset('Twibot-22') / 'graph.bin'
            graphs, _ = load_graphs(str(graph_path))
            graph = graphs[0]

            # Normalize label to 1-D long tensor
            if graph.ndata['label'].dim() > 1:
                graph.ndata['label'] = graph.ndata['label'].argmax(1)
            graph.ndata['label'] = graph.ndata['label'].long()

            # Attach edge_type tensor per edge type for downstream consumers
            for etype in graph.etypes:
                if 'edge_type' not in graph.edges[etype].data:
                    graph.edges[etype].data['edge_type'] = torch.zeros(
                        graph.num_edges(etype), dtype=torch.long)
        elif name == 'MGTAB':
            # Load from raw PyTorch files (PyG-style data)
            path = str(raw_dataset('MGTAB'))
            features = torch.load(osp.join(path, 'features.pt')).float()
            edge_index = torch.load(osp.join(path, 'edge_index.pt')).long()
            edge_type = torch.load(osp.join(path, 'edge_type.pt')).long()
            edge_weight_path = osp.join(path, 'edge_weight.pt')
            edge_weight = torch.load(edge_weight_path).float() if osp.exists(edge_weight_path) else None
            labels_bot = torch.load(osp.join(path, 'labels_bot.pt')).long()

            num_nodes = features.shape[0]
            src, dst = edge_index

            # Build heterograph with 7 relation types
            graph_data = {}
            for i in range(7):
                mask = (edge_type == i)
                if mask.sum() > 0:
                    graph_data[('_N', f'_E{i}', '_N')] = (src[mask], dst[mask])

            graph = heterograph(graph_data, num_nodes_dict={'_N': num_nodes})
            graph.ndata['feature'] = features
            graph.ndata['label'] = labels_bot

            # Add edge_type tensor
            for i in range(7):
                etype = f'_E{i}'
                if etype in graph.etypes:
                    mask = (edge_type == i)
                    graph.edges[etype].data['edge_type'] = torch.full(
                        (graph.num_edges(etype),), i, dtype=torch.long)
                    if edge_weight is not None:
                        graph.edges[etype].data['weight'] = edge_weight[mask].float()

            # Create train/val/test masks (70/20/10 split)
            train_idx = torch.arange(0, int(0.7 * num_nodes))
            val_idx = torch.arange(int(0.7 * num_nodes), int(0.9 * num_nodes))
            test_idx = torch.arange(int(0.9 * num_nodes), num_nodes)

            graph.ndata['train_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
            graph.ndata['val_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
            graph.ndata['test_mask'] = torch.zeros(num_nodes, dtype=torch.bool)
            graph.ndata['train_mask'][train_idx] = True
            graph.ndata['val_mask'][val_idx] = True
            graph.ndata['test_mask'][test_idx] = True

            save_graphs(str(cached_dataset('MGTAB')), [graph], {'labels': labels_bot})
        else:
            raise ValueError(f"Unknown dataset: {name}")

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph


def get_train_data(dataset_name):
    """
    Load the selected dataset and prepare it for pretraining.
    
    Args:
        dataset_name: Dataset name: 'Twibot-20', 'Twibot-22', 'tfinance', or 'tsocial'.
        
    Returns:
        graph: Processed graph object ready for pretraining.
    """
    # Load the selected dataset.
    print(f"Loading the {dataset_name} dataset...")
    dataset = Dataset(name=dataset_name)
    graph = dataset.graph
    
    # Normalize feature and label field names.
    if 'x' not in graph.ndata and 'feature' in graph.ndata:
        print("Copying 'feature' to 'x'")
        graph.ndata['x'] = graph.ndata['feature']
    
    if 'y' not in graph.ndata and 'label' in graph.ndata:
        print("Copying 'label' to 'y'")
        graph.ndata['y'] = graph.ndata['label']
    
    # Use all nodes as the pretraining index.
    graph.pre_trained_test_idx = torch.arange(0, graph.num_nodes())
    
    # Prepare dataset-specific attributes.
    if dataset_name == 'Twibot-20':
        path = str(raw_dataset('Twibot-20_raw'))
        graph.des_embedding = torch.load(osp.join(path, 'des_tensor.pt')).contiguous().float()
        graph.tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt')).contiguous().float()
        graph.num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt')).contiguous().float()
        graph.cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt')).contiguous().float()
    elif dataset_name == 'Twibot-22':
        # Twibot-22 graph.bin already contains des/tweet/num_prop/cat_prop embeddings in ndata
        for key in ['des_embedding', 'tweet_embedding', 'num_property_embedding', 'cat_property_embedding']:
            if key in graph.ndata:
                setattr(graph, key, graph.ndata[key].float())
    
    # Check for train/validation/test masks and create them if absent.
    if 'train_mask' not in graph.ndata:
        print(f"Creating training, validation, and test masks for {dataset_name}")
        # Create a random split (80% train, 10% validation, 10% test).
        indices = torch.randperm(graph.num_nodes())
        train_size = int(0.8 * graph.num_nodes())
        val_size = int(0.1 * graph.num_nodes())
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        # Create the corresponding masks.
        graph.ndata['train_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata['val_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata['test_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        
        graph.ndata['train_mask'][train_idx] = True
        graph.ndata['val_mask'][val_idx] = True
        graph.ndata['test_mask'][test_idx] = True
        
        # Save the indices.
        graph.train_idx = train_idx
        graph.val_idx = val_idx
        graph.test_idx = test_idx
    else:
        # Extract indices from the masks.
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        
        graph.train_idx = torch.nonzero(train_mask).squeeze()
        graph.val_idx = torch.nonzero(val_mask).squeeze()
        graph.test_idx = torch.nonzero(test_mask).squeeze()
    
    print(f"Nodes: {graph.num_nodes()}, edges: {graph.num_edges()}")
    print(f"Training: {len(graph.train_idx)}, validation: {len(graph.val_idx)}, test: {len(graph.test_idx)}")
    
    return graph
