import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
import random
import numpy as np
import pickle as pkl
from dataset import Dataset, get_train_data
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from mlp import MLP, MLP_Twibot
from sklearn.model_selection import train_test_split
import os
from paths import MLP_ROOT, PRETRAINED_ROOT

def train(model, data, args, seed, dataset_name):
    device = next(model.parameters()).device

    # Select features and preprocessing for the dataset.
    if dataset_name.startswith('Twibot'):
        des_embedding = data.des_embedding
        tweet_embedding = data.tweet_embedding
        num_property_embedding = data.num_property_embedding
        cat_property_embedding = data.cat_property_embedding
        edge_index = None
        edge_type = None
        labels = data.ndata['y']
        idx_train = data.train_idx
        idx_valid = data.val_idx
        idx_test = data.test_idx
    else:
        features = data.ndata['feature']
        labels = data.ndata['y']
        idx_train = data.train_idx
        idx_valid = data.val_idx
        idx_test = data.test_idx

    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)

    train_mask[idx_train] = True
    val_mask[idx_valid] = True
    test_mask[idx_test] = True

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    pred_y = None

    weight = 1.0
    if hasattr(data, 'weights'):
        weight = data.weights
    print('cross entropy weight: ', weight)

    no_improvement_count = 0
    embedding = None
    best_embedding = None
    best_state_dict = None

    for e in range(1, args.epoch+1):
        time_start_epoch = time.time()
        model.train()

        if dataset_name.startswith('Twibot'):
            logits, embedding = model(des_embedding, tweet_embedding,
                                     num_property_embedding, cat_property_embedding,
                                     edge_index, edge_type)
        else:
            logits, embedding = model(features)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask],
                               weight=torch.tensor([1., weight], device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask],
                                    weight=torch.tensor([1., weight], device=device))
        probs = logits.softmax(1)

        # sklearn needs CPU numpy
        labels_cpu = labels.cpu()
        probs_cpu = probs.detach().cpu()
        train_mask_cpu = train_mask.cpu()
        val_mask_cpu = val_mask.cpu()
        test_mask_cpu = test_mask.cpu()

        f1, thres = get_best_f1(labels_cpu[val_mask_cpu], probs_cpu[val_mask_cpu])
        preds = torch.zeros_like(labels)  # GPU tensor
        preds[probs[:, 1] > thres] = 1
        preds_cpu = preds.cpu()
        trec = recall_score(labels_cpu[test_mask_cpu].numpy(), preds_cpu[test_mask_cpu].numpy())
        tpre = precision_score(labels_cpu[test_mask_cpu].numpy(), preds_cpu[test_mask_cpu].numpy())
        tmf1 = f1_score(labels_cpu[test_mask_cpu].numpy(), preds_cpu[test_mask_cpu].numpy(), average='macro')
        tauc = roc_auc_score(labels_cpu[test_mask_cpu].numpy(), probs_cpu[test_mask_cpu][:, 1].numpy())

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            pred_y = probs_cpu  # keep on CPU for pickle
            best_embedding = embedding
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        time_end_epoch = time.time()
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, val_loss, f1, best_f1*100))
        print('time per epoch cost: ', time_end_epoch - time_start_epoch, 's')

        if no_improvement_count >= 50:
            print("No improvement in the last 50 epochs. training finished.")
            break

    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    os.makedirs(MLP_ROOT, exist_ok=True)
    model_path = MLP_ROOT / f'model_state_dict_{dataset_name}_seed{seed}.pth'
    if best_state_dict is not None:
        torch.save(best_state_dict, model_path)
    else:
        torch.save(model.state_dict(), model_path)
    return final_tmf1, final_tauc, best_embedding if best_embedding is not None else embedding, str(model_path)

def get_embeddings(model_path, dataset_name, data, in_feats=10, h_feats=10, num_classes=2, hidden_dim=64, device='cpu'):
    """
    Extract node embeddings from a saved model.
    
    Args:
        model_path: Path to the saved model.
        dataset_name: Dataset name.
        data: Graph data.
        in_feats: Input feature dimension.
        h_feats: Hidden feature dimension.
        num_classes: Number of classes.
        hidden_dim: Hidden dimension for the TwiBot model.
        device: Device used for inference.
    
    Returns:
        embeddings: Node embeddings.
    """
    # Select the model for the dataset.
    if dataset_name.startswith('Twibot'):
        # Use the TwiBot-specific model.
        model = MLP_Twibot(hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Read the input features.
        des_embedding = data.des_embedding
        tweet_embedding = data.tweet_embedding
        num_property_embedding = data.num_property_embedding
        cat_property_embedding = data.cat_property_embedding
        edge_index = None  # Read edge indices from the graph if needed.
        edge_type = None   # Read edge types from the graph if needed.
        
        # Extract the embeddings.
        with torch.no_grad():
            _, embeddings = model(des_embedding, tweet_embedding, 
                                 num_property_embedding, cat_property_embedding,
                                 edge_index, edge_type)
    else:
        # Use the standard model for other datasets.
        model = MLP(in_feats, h_feats, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Read the input features.
        features = data.ndata['feature']
        
        # Extract the embeddings.
        with torch.no_grad():
            _, embeddings = model(features)

    return embeddings

# threshold adjusting for best macro f1  (inputs: torch tensors, may be on GPU)
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    probs_np = probs[:, 1].detach().cpu().numpy() if probs.is_cuda else probs[:, 1].detach().numpy()
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels_np)
        preds[probs_np > thres] = 1
        mf1 = f1_score(labels_np, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_training(args, seed):
    """Run training once and save the model and embeddings."""
    print(f"\n==== Starting training with seed {seed} ====")
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset_name = args.dataset

    # Load the data before constructing the model.
    data = get_train_data(dataset_name)
    data = data.to(device)  # Move DGL graph to GPU

    # Select model parameters for the dataset.
    if dataset_name.startswith('Twibot'):
        # Use the TwiBot-specific model parameters.
        hidden_dim = 128
        model = MLP_Twibot(hidden_dim=hidden_dim)
    else:
        # Use the standard model parameters for other datasets.
        in_feats = data.ndata['feature'].shape[1]
        h_feats = 10
        num_classes = 2
        model = MLP(in_feats, h_feats, num_classes)

    model = model.to(device)

    # Train the model.
    time_all_start = time.time()
    final_tmf1, final_tauc, embeddings, model_path = train(model, data, args, seed, dataset_name)
    time_all_end = time.time()
    print('Total pretraining time:', time_all_end - time_all_start, 'seconds')
    
    # Extract embeddings from the checkpoint if training did not return them.
    if embeddings is None:
        if dataset_name.startswith('Twibot'):
            embeddings = get_embeddings(model_path, dataset_name, data, hidden_dim=128, device=device)
        else:
            embeddings = get_embeddings(model_path, dataset_name, data, device=device)

    # Save embedding on CPU for portability
    embeddings_cpu = embeddings.cpu() if embeddings.is_cuda else embeddings
    print(f"Embedding shape: {embeddings_cpu.shape}")

    # Create the output directory if needed.
    os.makedirs(PRETRAINED_ROOT, exist_ok=True)

    # Save embeddings using the dataset and seed naming convention.
    embedding_file_name = f'{dataset_name}_pretrained_embedding_{seed}'
    output_path = PRETRAINED_ROOT / f'{embedding_file_name}.pt'
    torch.save(embeddings_cpu, output_path)
    print(f"Saved embeddings to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument("--dataset", type=str, default="tsocial", 
                        help="Dataset for this model (tsocial, Twibot-20, Twibot-22, tfinance)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=3, help="Number of runs with different seeds")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--del_ratio", type=float, default=0.0, help="Fraction of nodes to remove when saving model predictions")
    parser.add_argument("--homo", type=int, default=1, help="Whether to use a homogeneous graph (affects the output filename)")

    args = parser.parse_args()
    print(args)

    # Train once per seed.
    for seed in range(args.start_seed, args.start_seed + args.run):
        run_training(args, seed)
