# ppr.py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numba
from sklearn.metrics.pairwise import cosine_similarity
import dgl
import os
from paths import cached_dataset

######################
# Core PPR computation (Numba-optimized)
######################

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, N):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    
    # Use a Numba-compatible dictionary.
    p = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.float32
    )
    r = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.float32
    )
    
    p[inode] = f32_0
    r[inode] = alpha
    q = [inode]

    while q:
        unode = q.pop()
        res = r.get(unode, f32_0)
        p[unode] = p.get(unode, f32_0) + res
        r[unode] = f32_0

        # Read the bounds of the current CSR row.
        start = indptr[unode]
        end = indptr[unode + 1]
        for idx in range(start, end):
            vnode = indices[idx]
            
            # Handle nodes with degree zero.
            if deg[unode] < 1e-5:
                _val = (1 - alpha) * res / N
            else:
                _val = (1 - alpha) * res / deg[unode]
            
            r[vnode] = r.get(vnode, f32_0) + _val
            
            # Check the residual against the threshold.
            if r[vnode] >= alpha_eps * max(deg[vnode], 1.0):
                if vnode not in q:
                    q.append(vnode)

    # Specify the array types explicitly.
    keys = np.array(list(p.keys()), dtype=np.int64)
    values = np.array(list(p.values()), dtype=np.float32)
    return keys, values  # Return separate arrays for indices and scores.

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, N):
    js = [np.empty(0, dtype=np.int64) for _ in range(len(nodes))]
    vals = [np.empty(0, dtype=np.float32) for _ in range(len(nodes))]
    
    for i in numba.prange(len(nodes)):
        node = nodes[i]
        
        # Handle nodes with degree zero.
        if deg[node] < 1e-5:
            js[i] = np.array([node], dtype=np.int64)
            vals[i] = np.array([1.0], dtype=np.float32)
            continue
            
        # Compute PPR.
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon, N)
        
        # Handle empty results.
        if len(j) == 0:
            js[i] = np.array([node], dtype=np.int64)
            vals[i] = np.array([1.0], dtype=np.float32)
            continue
            
        # Truncate and normalize the scores.
        idx_topk = np.argsort(val)[-topk:]
        total = np.sum(val[idx_topk])
        if total < 1e-12:
            js[i] = np.array([node], dtype=np.int64)
            vals[i] = np.array([1.0], dtype=np.float32)
        else:
            js[i] = j[idx_topk]
            vals[i] = val[idx_topk] / total
            
    return js, vals

######################
# Sparse matrix interface
######################

def construct_sparse(neighbors, weights, shape):
    """Build a sparse matrix in COO format."""
    row = np.repeat(np.arange(len(neighbors)), [len(n) for n in neighbors])
    col = np.concatenate(neighbors)
    data = np.concatenate(weights)
    return sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)

def ppr_topk(adj_matrix, alpha=0.85, epsilon=1e-5, nodes=None, topk=20):
    """Compute top-k PPR scores for the selected nodes."""
    adj_matrix = adj_matrix.tocsr()
    N = adj_matrix.shape[0]
    
    # Cast inputs to Numba-compatible types.
    indptr = adj_matrix.indptr.astype(np.int32)
    indices = adj_matrix.indices.astype(np.int32)
    deg = np.maximum(adj_matrix.sum(1).A1, 1e-12).astype(np.float32)
    
    # Prepare the node indices.
    if nodes is None:
        nodes = np.arange(N, dtype=np.int32)
    else:
        nodes = nodes.astype(np.int32)
    
    # Compute PPR in parallel.
    neighbors, weights = calc_ppr_topk_parallel(
        indptr, indices, deg,
        np.float32(alpha),
        np.float32(epsilon),
        nodes,
        topk,
        np.int32(N)
    )
    
    return construct_sparse(neighbors, weights, (len(nodes), N))

######################
# Multi-hop PPR computation and fusion
######################

def calc_multi_hop_ppr(adj_matrix, alpha=0.85, epsilon=1e-5, nodes=None, topk=20, 
                       hops=(1, 2, 3), hop_weights=(0.6, 0.3, 0.1)):
    """Compute PPR at multiple hop counts and combine the results.
    
    Args:
        adj_matrix: Sparse adjacency matrix.
        alpha: PPR damping parameter that controls the random walk's restart probability.
        epsilon: Convergence threshold.
        nodes: Indices of nodes to evaluate. Defaults to all nodes.
        topk: Number of neighbors to retain per node.
        hops: Hop counts to evaluate, for example (1, 2, 3).
        hop_weights: One weight per hop count, in the same order as hops.
        
    Returns:
        Combined sparse PPR matrix.
    """
    assert len(hops) == len(hop_weights), "hops and hop_weights must have the same length"
    assert sum(hop_weights) <= 1.0, "Hop weights must sum to at most 1.0"
    
    if nodes is None:
        nodes = np.arange(adj_matrix.shape[0], dtype=np.int32)
    else:
        nodes = np.asarray(nodes, dtype=np.int32)
    
    # Store the PPR result for each hop count.
    hop_ppr_results = []
    
    # Compute PPR for each hop count.
    for hop, weight in zip(hops, hop_weights):
        # Adjust the supplied alpha for the current hop count.
        hop_alpha = alpha ** hop  # Other hop-dependent schedules are possible.
        
        if hop == 1:
            # Use the original adjacency matrix for one hop.
            ppr_matrix = ppr_topk(adj_matrix, alpha=hop_alpha, epsilon=epsilon, 
                                 nodes=nodes, topk=topk)
        else:
            # Use powers of the adjacency matrix for multiple hops.
            hop_adj = adj_matrix.copy()
            for _ in range(hop-1):
                hop_adj = hop_adj.dot(adj_matrix)
            
            # Compute PPR on the multi-hop adjacency matrix.
            ppr_matrix = ppr_topk(hop_adj, alpha=hop_alpha, epsilon=epsilon, 
                                 nodes=nodes, topk=topk)
        
        hop_ppr_results.append((ppr_matrix, weight))
    
    # Combine the results across hop counts.
    return fuse_multi_hop_ppr(hop_ppr_results, nodes, adj_matrix.shape[0], topk)

def fuse_multi_hop_ppr(hop_ppr_results, nodes, N, topk):
    """Combine PPR scores across hop counts.
    
    Args:
        hop_ppr_results: List of (PPR matrix, weight) pairs.
        nodes: Node indices.
        N: Total number of nodes in the graph.
        topk: Number of neighbors to retain per node.
        
    Returns:
        Combined sparse PPR matrix.
    """
    # Initialize the output arrays.
    fused_neighbors = [[] for _ in range(len(nodes))]
    fused_weights = [[] for _ in range(len(nodes))]
    
    # Collect PPR contributions across hop counts for each node.
    node_contributions = {}  # (node_idx, neighbor_idx) -> accumulated_weight
    
    # Process the PPR matrix for each hop count.
    for ppr_matrix, hop_weight in hop_ppr_results:
        # Extract each node and its neighbors.
        for i, node_idx in enumerate(nodes):
            # Read the current node's row.
            row = ppr_matrix.getrow(i)
            neighbors = row.indices
            weights = row.data * hop_weight  # Apply the weight for this hop count.
            
            # Accumulate weighted contributions.
            for neighbor, weight in zip(neighbors, weights):
                key = (i, neighbor)
                node_contributions[key] = node_contributions.get(key, 0.0) + weight
    
    # Select the top-k neighbors for each node.
    for i, node_idx in enumerate(nodes):
        # Collect the current node's neighbor scores.
        node_neighbors = []
        node_weights = []
        
        for (ni, neighbor), weight in node_contributions.items():
            if ni == i:  # Match the current node.
                node_neighbors.append(neighbor)
                node_weights.append(weight)
        
        if not node_neighbors:
            # Add a self-loop when the node has no neighbors.
            fused_neighbors[i] = np.array([node_idx], dtype=np.int64)
            fused_weights[i] = np.array([1.0], dtype=np.float32)
            continue
        
        # Convert to NumPy arrays for sorting.
        node_neighbors = np.array(node_neighbors)
        node_weights = np.array(node_weights)
        
        # Select the k neighbors with the highest weights.
        if len(node_neighbors) > topk:
            top_indices = np.argsort(node_weights)[-topk:]
            node_neighbors = node_neighbors[top_indices]
            node_weights = node_weights[top_indices]
        
        # Normalize the selected weights.
        total_weight = np.sum(node_weights)
        if total_weight > 0:
            node_weights = node_weights / total_weight
        else:
            # Use uniform weights if the sum is zero.
            node_weights = np.ones_like(node_weights) / len(node_weights)
        
        fused_neighbors[i] = node_neighbors
        fused_weights[i] = node_weights
    
    # Build the combined sparse matrix.
    return construct_sparse(fused_neighbors, fused_weights, (len(nodes), N))

######################
# Heterogeneous graph processing
######################

def get_sparse_matrices(data, directed=True):
    """Build an adjacency matrix for each edge type."""
    num_nodes = data.num_nodes()
    adj_dict = {}
    
    for etype in data.etypes:
        src, dst = data.edges(etype=etype)
        src = src.cpu().numpy().astype(np.int32)
        dst = dst.cpu().numpy().astype(np.int32)
        
        adj = sp.coo_matrix(
            (np.ones_like(src), (src, dst)),
            shape=(num_nodes, num_nodes)
        )
        if not directed:
            adj = adj + adj.T
        adj_dict[etype] = adj.tocsr()
    
    return adj_dict

def get_biased_ppr_matrix(data, ppr_matrix, topk, alpha=0.15):
    """Combine PPR scores with feature similarity, using GPU batching when available.

    The GPU path batches cosine similarity with PyTorch in place of a per-node scikit-learn loop.
    On the 5.78-million-node T-Social graph, this reduced runtime from hours to tens of seconds.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move features to the selected device and L2-normalize once.
    features_np = data.ndata['pretrained_results'].cpu().detach().numpy()
    features_gpu = torch.from_numpy(features_np).float().to(device)
    features_gpu = F.normalize(features_gpu, p=2, dim=1)  # [N, d]

    ppr_csr = ppr_matrix.tocsr()
    N = ppr_csr.shape[0]

    if device.type == 'cuda':
        # GPU path: batch similarity calculations across all edges.
        # Extract (source, destination, PPR score) triples from COO.
        ppr_coo = ppr_csr.tocoo()
        src_all = torch.from_numpy(ppr_coo.row).long().to(device)
        dst_all = torch.from_numpy(ppr_coo.col).long().to(device)
        ppr_all = torch.from_numpy(ppr_coo.data).float().to(device)

        # Remove self-loops.
        keep = src_all != dst_all
        src_all = src_all[keep]
        dst_all = dst_all[keep]
        ppr_all = ppr_all[keep]

        # Compute cosine similarity as the dot product of normalized features.
        sim_all = (features_gpu[src_all] * features_gpu[dst_all]).sum(dim=1)

        # Combine feature similarity and PPR scores.
        combined = alpha * sim_all + (1 - alpha) * ppr_all

        # Move the results back to NumPy on the CPU.
        src_np = src_all.cpu().numpy()
        dst_np = dst_all.cpu().numpy()
        combined_np = combined.cpu().numpy()

        # Group by source and retain the top-k neighbors per group.
        rows, cols, vals = _select_topk_per_node(src_np, dst_np, combined_np, N, topk)
    else:
        # CPU fallback: compute similarity one node at a time.
        rows, cols, vals = [], [], []
        for i in range(N):
            row = ppr_csr.getrow(i)
            if row.nnz == 0:
                continue
            mask = row.indices != i
            neighbors = row.indices[mask]
            ppr_vals = row.data[mask]
            if len(neighbors) == 0:
                continue
            sim = cosine_similarity(
                features_np[i].reshape(1, -1),
                features_np[neighbors]
            ).flatten()
            combined_i = alpha * sim + (1 - alpha) * ppr_vals
            combined_i /= combined_i.sum() + 1e-12
            topk_idx = np.argsort(combined_i)[-topk:]
            rows.extend([i] * len(topk_idx))
            cols.extend(neighbors[topk_idx])
            vals.extend(combined_i[topk_idx])

    return sp.coo_matrix((vals, (rows, cols)), shape=ppr_matrix.shape)


def _select_topk_per_node(src_np, dst_np, combined_np, N, topk):
    """Group edges by source and keep the top-k neighbors by combined score.
    Sort by source with NumPy, then rank neighbors within each group.
    """
    # Sort edges by source node.
    order = np.argsort(src_np)
    src_sorted = src_np[order]
    dst_sorted = dst_np[order]
    combined_sorted = combined_np[order]

    # Find the group boundaries for each source node.
    # Use np.unique to get each group's starting offset.
    unique_src, start_idx, counts = np.unique(src_sorted, return_index=True, return_counts=True)

    rows_out = []
    cols_out = []
    vals_out = []

    for k in range(len(unique_src)):
        s = start_idx[k]
        c = counts[k]
        if c <= topk:
            # Retain all neighbors in this group.
            rows_out.append(np.full(c, unique_src[k], dtype=np.int64))
            cols_out.append(dst_sorted[s:s + c])
            sel_vals = combined_sorted[s:s + c]
        else:
            # Select the top-k neighbors.
            topk_local = np.argpartition(combined_sorted[s:s + c], -topk)[-topk:]
            topk_local = topk_local[np.argsort(combined_sorted[s:s + c][topk_local])[::-1]]
            rows_out.append(np.full(topk, unique_src[k], dtype=np.int64))
            cols_out.append(dst_sorted[s:s + c][topk_local])
            sel_vals = combined_sorted[s:s + c][topk_local]

        # Normalize the selected scores.
        total = sel_vals.sum()
        if total > 1e-12:
            vals_out.append(sel_vals / total)
        else:
            vals_out.append(sel_vals)

    rows = np.concatenate(rows_out) if rows_out else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_out) if cols_out else np.array([], dtype=np.int64)
    vals = np.concatenate(vals_out) if vals_out else np.array([], dtype=np.float32)

    return rows.astype(np.int64), cols.astype(np.int64), vals.astype(np.float32)

######################
# PPR workflow
######################

def topk_ppr_matrix(data, alpha=0.85, epsilon=1e-5, topk=20, feature_alpha=0.85,
                   use_multi_hop=True, hops=(1, 2, 3), hop_weights=(0.6, 0.3, 0.1)):
    """Compute a feature-biased PPR matrix for each edge type.
    
    Args:
        data: DGL graph.
        alpha: PPR damping parameter.
        epsilon: Convergence threshold.
        topk: Number of neighbors to retain per node.
        feature_alpha: Weight for feature similarity.
        use_multi_hop: Whether to use multi-hop PPR.
        hops: Hop counts to evaluate when use_multi_hop is True.
        hop_weights: Weight for each hop count when use_multi_hop is True.
    """
    adj_dict = get_sparse_matrices(data)
    ppr_dict = {}
    
    for etype, adj in adj_dict.items():
        if use_multi_hop:
            # Compute multi-hop PPR.
            base_ppr = calc_multi_hop_ppr(
                adj, alpha=alpha, epsilon=epsilon,
                nodes=np.arange(adj.shape[0], dtype=np.int32),
                topk=topk, hops=hops, hop_weights=hop_weights
            )
        else:
            # Compute standard PPR.
            base_ppr = ppr_topk(
                adj, alpha=alpha, epsilon=epsilon,
                nodes=np.arange(adj.shape[0], dtype=np.int32),
                topk=topk
            )
        
        # Weight the PPR scores using feature similarity.
        biased_ppr = get_biased_ppr_matrix(data, base_ppr.tocsr(), topk, alpha=feature_alpha)
        ppr_dict[etype] = biased_ppr.tocsr()
    
    return ppr_dict

def add_edges_and_save_graph(data, pretrained_embeddings, topk=20, dataset_name=None, 
                           use_multi_hop=True, hops=(1, 2, 3), hop_weights=(0.6, 0.3, 0.1),
                           alpha=0.85, epsilon=1e-5, feature_alpha=0.85):
    """Add PPR edges and save the augmented graph.
    
    Args:
        data: DGL graph.
        pretrained_embeddings: Pretrained node embeddings.
        topk: Number of neighbors to retain per node during PPR.
        dataset_name: Dataset name.
        use_multi_hop: Whether to use multi-hop PPR.
        hops: Hop counts to evaluate when use_multi_hop is True.
        hop_weights: Weight for each hop count when use_multi_hop is True.
        alpha: PPR damping parameter that controls the random walk's restart probability.
        epsilon: Convergence threshold.
        feature_alpha: Weight for feature similarity.
    """
    # Attach the pretrained embeddings to the graph.
    data.ndata['pretrained_results'] = pretrained_embeddings

    # Compute the PPR edges to add.
    ppr_dict = topk_ppr_matrix(
        data, 
        topk=topk, 
        use_multi_hop=use_multi_hop, 
        hops=hops, 
        hop_weights=hop_weights,
        alpha=alpha,
        epsilon=epsilon,
        feature_alpha=feature_alpha
    )

    # Inspect the graph's node and edge types.
    if len(data.ntypes) == 1:  # Single node type.
        node_type = data.ntypes[0]
        new_edges = {}
        
        # Preserve the original edges.
        if len(data.etypes) == 1:  # Single edge type.
            src, dst = data.edges()
            new_edges[(node_type, 'original', node_type)] = (src, dst)
        else:  # Multiple edge types.
            for orig_etype in data.etypes:
                src, dst = data.edges(etype=orig_etype)
                new_edges[(node_type, orig_etype, node_type)] = (src, dst)
        
        # Add PPR edges.
        for orig_etype, ppr_mat in ppr_dict.items():
            rows, cols = ppr_mat.nonzero()
            new_edges[(node_type, f'ppr_{orig_etype}{"_multi" if use_multi_hop else ""}', node_type)] = (
                torch.from_numpy(rows).long(),
                torch.from_numpy(cols).long()
            )
        
        # Construct the augmented graph.
        num_nodes_dict = {node_type: data.num_nodes()}
        new_graph = dgl.heterograph(new_edges, num_nodes_dict=num_nodes_dict)
        
        # Copy the node features.
        for key in data.ndata:
            new_graph.ndata[key] = data.ndata[key].clone()
            
    else:  # Multiple node types.
        raise NotImplementedError("PPR augmentation requires a graph with a single node type")
    
    # Save the augmented graph.
    if dataset_name is None:
        dataset_name = getattr(data, 'dataset_name', 'unknown')
    hop_suffix = '_multi' if use_multi_hop else ''
    save_dir = cached_dataset(f'{dataset_name}_new_topk{topk}{hop_suffix}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / 'graph.bin'
    dgl.save_graphs(str(save_path), [new_graph])
    print(f'Saved the augmented graph to {save_path}')
    
    # Return the augmented graph and the new PPR edge types.
    return new_graph, [etype for etype in new_graph.etypes if 'ppr' in etype]
