import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from math import comb as math_comb
import math  # Import the math module.
import numpy as np

def orthogonalize_filters(alphas):
    if torch.isnan(alphas).any() or torch.isinf(alphas).any():
        print("Input matrix contains nan or inf values!")
        alphas = torch.nan_to_num(alphas, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep the returned shape as [num_scales, K+1]. Full row orthogonality is
    # only possible when num_scales <= K+1; otherwise use the closest
    # shape-preserving polar projection instead of dropping scale rows.
    reg = 1e-6 * torch.eye(alphas.size(0), alphas.size(1), device=alphas.device)
    alphas = alphas + reg

    num_scales, basis_dim = alphas.shape
    if num_scales <= basis_dim:
        Q, _ = torch.linalg.qr(alphas.T)
        return Q.T

    U, _, Vh = torch.linalg.svd(alphas, full_matrices=False)
    return U @ Vh

class OptimizedBernsteinConv(nn.Module):
    def __init__(self, in_feats, out_feats, K=4, num_scales=5,
                 activation=F.gelu, bias=False, etype=None, use_edge_weight=False):
        super().__init__()
        self.K = K
        self.num_scales = num_scales
        self.etype = etype
        self.use_edge_weight = use_edge_weight
        self.activation = activation

        # Per-scale linear feature transformation: W_{r,s} in Eq.(9)
        self.W_s = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=bias)
            for _ in range(num_scales)
        ])

        # Window combination weights: β_{r,s} in Eq.(13)
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def _compute_coeffs_optimized(self, K, omega, sigma):
        device = omega.device
        n_points = 32

        if not hasattr(self, 'gl_points') or self.gl_points.shape[0] != n_points:
            points_np, weights_np = np.polynomial.legendre.leggauss(n_points)
            points = torch.tensor(points_np, dtype=torch.float32)
            weights = torch.tensor(weights_np, dtype=torch.float32)
            self.register_buffer('gl_points', points * 1.0 + 1.0)  # [-1,1] → [0,2]
            self.register_buffer('gl_weights', weights * 1.0)

        all_coeffs = []
        for s in range(self.num_scales):
            x = self.gl_points.unsqueeze(0).to(device)
            gw = self.gl_weights.to(device)

            sigma_s = torch.clamp(sigma[s], min=0.1)
            g = torch.exp(-0.5 * ((x - omega[s]) / sigma_s) ** 2)

            coeffs = []
            for k in range(K + 1):
                binom_coeff = math_comb(K, k)
                B = binom_coeff * ((x / 2) ** k) * ((1 - x / 2) ** (K - k))
                c_k = torch.sum(g * B * gw)
                coeffs.append(c_k)
            all_coeffs.append(torch.stack(coeffs))

        return torch.stack(all_coeffs)

    def _laplacian(self, feat, inv_degree, graph):
        with graph.local_scope():
            graph.ndata['h'] = feat
            if self.use_edge_weight and 'weight' in graph.edges[self.etype].data:
                weight = graph.edges[self.etype].data['weight'].to(feat.device, feat.dtype)
                graph.edges[self.etype].data['w'] = weight.unsqueeze(-1)
                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h_sum'),
                                 etype=self.etype)
            else:
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h_sum'),
                                 etype=self.etype)
            return inv_degree * (feat - graph.ndata['h_sum'])

    def compute_alpha(self, coeffs):
        alphas = []
        for c in coeffs:
            alpha = torch.zeros_like(c)
            for m in range(self.K + 1):
                for k in range(m + 1):
                    beta = (math_comb(self.K, k) *
                            math_comb(self.K - k, m - k) *
                            ((-1) ** (m - k)) *
                            (0.5 ** m))
                    alpha[m] = alpha[m] + c[k] * beta
            alphas.append(alpha)
        return orthogonalize_filters(torch.stack(alphas))

    def forward(self, graph, feat, omega, sigma):
        """
        Args:
            graph: DGL graph
            feat: node features [N, d_in]
            omega: Gaussian window centers [num_scales] from Homophily-Aware MLP
            sigma: Gaussian window bandwidths [num_scales] from Homophily-Aware MLP
        Returns:
            h: filtered features [N, out_feats]
            omega_effective: weighted avg ω for L_freq
        """
        with graph.local_scope():
            if self.use_edge_weight and 'weight' in graph.edges[self.etype].data:
                with graph.local_scope():
                    weight = graph.edges[self.etype].data['weight'].to(feat.device, feat.dtype)
                    graph.edges[self.etype].data['w_deg'] = weight
                    graph.update_all(fn.copy_e('w_deg', 'm'),
                                     fn.sum('m', 'w_deg_sum'),
                                     etype=self.etype)
                    degree = graph.ndata['w_deg_sum'].float()
            else:
                degree = graph.in_degrees(etype=self.etype).float()
            inv_degree = torch.where(degree > 0, 1.0 / degree.clamp(min=1e-6), torch.ones_like(degree))
            inv_degree = inv_degree.unsqueeze(-1).to(feat.device)

            coeffs = self._compute_coeffs_optimized(self.K, omega, sigma).to(feat.device)
            alphas = self.compute_alpha(coeffs)  # [num_scales, K+1]

            # Precompute Laplacian powers T_m = L^m * feat (shared across scales)
            T = [feat]
            for m in range(1, self.K + 1):
                T.append(self._laplacian(T[-1], inv_degree, graph))

            # Eq.(9): Z_{r,s} = Σ_k c_{r,s,k} P_k(L_r) X W_{r,s}
            # = Σ_m α_{r,s,m} L_r^m X W_{r,s}
            window_weights = F.softmax(self.beta, dim=0)  # Eq.(13)

            h = torch.zeros(feat.shape[0], self.W_s[0].out_features,
                            device=feat.device, dtype=feat.dtype)
            for s in range(self.num_scales):
                scale_feat = torch.zeros_like(feat)
                for m in range(self.K + 1):
                    scale_feat += alphas[s, m] * T[m]
                scale_h = self.W_s[s](scale_feat)
                h += window_weights[s] * scale_h

            # Effective center frequency for L_freq
            omega_effective = torch.sum(window_weights * omega)

            return self.activation(h), omega_effective

class EnhancedBernsteinNet(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, K=4, num_scales=5,
                 num_layers=2, dropout_rate=0.3, focal_gamma=2.0, focal_alpha=0.25,
                 lambda_freq=0.2, use_pretrained_skip=True, etypes=None,
                 relation_fusion='mean', homophily_temperature=8.0,
                 freq_loss_weighting='uniform', use_edge_weight=False,
                 final_residual_weight=0.0):
        super().__init__()
        self.g = graph
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.lambda_freq = lambda_freq
        self.use_pretrained_skip = use_pretrained_skip
        self.relation_fusion = relation_fusion
        self.homophily_temperature = homophily_temperature
        self.freq_loss_weighting = freq_loss_weighting
        self.use_edge_weight = use_edge_weight
        self.final_residual_weight = final_residual_weight

        self.is_twibot = ('des_embedding' in graph.ndata and 'tweet_embedding' in graph.ndata)

        if self.is_twibot:
            self.des_linear = nn.Linear(graph.ndata['des_embedding'].shape[1], h_feats // 4)
            self.tweet_linear = nn.Linear(graph.ndata['tweet_embedding'].shape[1], h_feats // 4)
            self.num_prop_linear = nn.Linear(graph.ndata['num_property_embedding'].shape[1], h_feats // 4)
            self.cat_prop_linear = nn.Linear(graph.ndata['cat_property_embedding'].shape[1], h_feats // 4)
            self.feature_fusion = nn.Sequential(
                nn.Linear(h_feats, h_feats),
                nn.LayerNorm(h_feats),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            if use_pretrained_skip and 'feature' in graph.ndata:
                self.pretrained_linear = nn.Sequential(
                    nn.Linear(in_feats, h_feats),
                    nn.LayerNorm(h_feats),
                    nn.GELU()
                )
            else:
                self.pretrained_linear = None
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_feats, h_feats),
                nn.LayerNorm(h_feats),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )

        if etypes is None:
            self.etypes = list(graph.etypes)
        else:
            available_etypes = set(graph.etypes)
            self.etypes = [etype for etype in etypes if etype in available_etypes]
            if not self.etypes:
                raise ValueError(f'No requested edge types found in graph: {etypes}')

        # Initial equally-spaced positions for Gaussian windows in [0, 2]
        self.register_buffer('init_omega', torch.linspace(0.1, 1.9, num_scales))

        # Homophily-Aware MLPs per edge type: Eq.(8)
        # MLP_ω: [ω̄(h_r), s_norm] → Δω  (offset from init position)
        # MLP_σ: [ω̄(h_r), s_norm] → σ   (bandwidth, positive)
        self.mlp_omega = nn.ModuleDict({
            etype: nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
            for etype in self.etypes
        })
        self.mlp_sigma = nn.ModuleDict({
            etype: nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
            for etype in self.etypes
        })

        # Multi-layer relation-specific convs
        self.convs = nn.ModuleList([
            nn.ModuleList([
                OptimizedBernsteinConv(h_feats, h_feats, K=K, num_scales=num_scales,
                                       etype=etype, use_edge_weight=use_edge_weight)
                for etype in self.etypes
            ])
            for _ in range(num_layers)
        ])

        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(h_feats) for _ in range(num_layers)
        ])

        # Global relation-level semantic attention (HAN-inspired)
        self.sem_W = nn.Parameter(torch.Tensor(h_feats, h_feats // 2))
        self.sem_b = nn.Parameter(torch.Tensor(h_feats // 2))
        self.sem_q = nn.Parameter(torch.Tensor(h_feats // 2, 1))
        self._reset_sem_params()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.LayerNorm(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, num_classes)
        )

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Homophily target (set via set_homophily)
        self._homophily = {}  # kept for external inspection
        self._omega_target = {}

    def _reset_sem_params(self):
        stdv = 1. / math.sqrt(self.sem_W.size(1))
        self.sem_W.data.uniform_(-stdv, stdv)
        self.sem_b.data.uniform_(-stdv, stdv)
        self.sem_q.data.uniform_(-stdv, stdv)

    def set_homophily(self, labels, train_mask):
        """Compute homophily ratio h_r and target ω̄(h_r) = 2*(1-h_r) per edge type."""
        with torch.no_grad():
            for etype in self.etypes:
                src, dst = self.g.edges(etype=etype)
                mask = train_mask[src] & train_mask[dst]
                if mask.sum() < 2:
                    h_r = 0.5
                else:
                    h_r = (labels[src[mask]] == labels[dst[mask]]).float().mean().item()
                self._homophily[etype] = h_r
                self._omega_target[etype] = 2.0 * (1.0 - h_r)

    def _compute_omega_sigma(self, etype, device):
        """Generate ω and σ for all scales via Homophily-Aware MLPs. Eq.(8)"""
        omega_bar = self._omega_target.get(etype, 1.0)
        margin = 0.3  # constrain ω_s within [init - margin, init + margin]
        omega_list, sigma_list = [], []
        for s in range(self.num_scales):
            s_norm = s / max(1, self.num_scales - 1)
            inp = torch.tensor([omega_bar, s_norm], dtype=torch.float32, device=device)

            delta = margin * torch.tanh(self.mlp_omega[etype](inp).squeeze())
            omega_s = torch.clamp(self.init_omega[s] + delta, 0.05, 1.95)

            sigma_s = torch.clamp(F.softplus(self.mlp_sigma[etype](inp).squeeze()) + 0.1,
                                  0.1, 2.0)

            omega_list.append(omega_s)
            sigma_list.append(sigma_s)
        return torch.stack(omega_list), torch.stack(sigma_list)

    def _semantic_fusion(self, h_list):
        """Global relation-level semantic attention. Returns fused [N, d]."""
        scores = []
        for z_r in h_list:
            w = torch.tanh(z_r @ self.sem_W + self.sem_b)  # [N, hid]
            s = (w @ self.sem_q).mean()  # scalar
            scores.append(s)
        scores = torch.stack(scores)  # [R]
        alpha = F.softmax(scores, dim=0)
        h = sum(alpha[i] * h_list[i] for i in range(len(h_list)))
        return h

    def _homophily_fusion(self, h_list, device):
        weights = []
        for etype in self.etypes:
            h_r = self._homophily.get(etype, 0.5)
            weights.append(h_r)
        weights = torch.tensor(weights, dtype=h_list[0].dtype, device=device)
        weights = F.softmax(self.homophily_temperature * weights, dim=0)
        return sum(weights[i] * h_list[i] for i in range(len(h_list)))

    def _edge_weight_fusion(self, h_list, device):
        weights = []
        for etype in self.etypes:
            if 'weight' in self.g.edges[etype].data:
                weights.append(float(self.g.edges[etype].data['weight'].float().mean().item()))
            else:
                weights.append(1.0)
        weights = torch.tensor(weights, dtype=h_list[0].dtype, device=device)
        weights = weights / weights.mean().clamp(min=1e-6)
        weights = F.softmax(self.homophily_temperature * weights, dim=0)
        return sum(weights[i] * h_list[i] for i in range(len(h_list)))

    def _combine_relations(self, h_etypes, final=False):
        if len(h_etypes) == 1:
            return h_etypes[0]
        if self.relation_fusion == 'semantic' or (final and self.relation_fusion == 'mean_final_semantic'):
            return self._semantic_fusion(h_etypes)
        if self.relation_fusion == 'homophily':
            return self._homophily_fusion(h_etypes, h_etypes[0].device)
        if self.relation_fusion == 'edge_weight':
            return self._edge_weight_fusion(h_etypes, h_etypes[0].device)
        return torch.mean(torch.stack(h_etypes), dim=0)

    def forward(self, in_feat, return_embedding=False):
        # Feature transform
        if self.is_twibot:
            des_feat = self.des_linear(self.g.ndata['des_embedding'])
            tweet_feat = self.tweet_linear(self.g.ndata['tweet_embedding'])
            num_prop_feat = self.num_prop_linear(self.g.ndata['num_property_embedding'])
            cat_prop_feat = self.cat_prop_linear(self.g.ndata['cat_property_embedding'])
            h = torch.cat([des_feat, tweet_feat, num_prop_feat, cat_prop_feat], dim=1)
            h = self.feature_fusion(h)
            if self.pretrained_linear is not None:
                h = h + self.pretrained_linear(self.g.ndata['feature'])
        else:
            h = self.linear(in_feat)

        self._omega_effs = []  # for L_freq
        final_h_etypes = None

        for layer_idx in range(self.num_layers):
            h_etypes = []
            for etype_idx, (conv, etype) in enumerate(zip(self.convs[layer_idx], self.etypes)):
                omega, sigma = self._compute_omega_sigma(etype, h.device)
                z_r, omega_eff = conv(self.g, h, omega, sigma)
                h_etypes.append(z_r)
                self._omega_effs.append(omega_eff)

            combined = self._combine_relations(h_etypes)

            h = F.gelu(self.layer_norms[layer_idx](h + combined))
            final_h_etypes = h_etypes

        if final_h_etypes is not None and len(final_h_etypes) > 1:
            fused_h = self._combine_relations(final_h_etypes, final=True)
            if self.final_residual_weight > 0.0:
                residual_weight = min(max(self.final_residual_weight, 0.0), 1.0)
                h = residual_weight * h + (1.0 - residual_weight) * fused_h
            else:
                h = fused_h

        if return_embedding:
            return h
        return self.classifier(h)

    def get_freq_loss(self):
        """L_freq = 1/|R| * Σ_r (ω̂_r - ω̄(h_r))²  Eq.(11)"""
        if not self._omega_effs:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        num_relations = len(self.etypes)
        layers_per_rel = len(self._omega_effs) // num_relations
        loss = 0.0
        for i, etype in enumerate(self.etypes):
            omega_bar = self._omega_target.get(etype, 1.0)
            if self.freq_loss_weighting == 'homophily':
                rel_weight = self._homophily.get(etype, 0.5)
            elif self.freq_loss_weighting == 'edge_weight' and 'weight' in self.g.edges[etype].data:
                rel_weight = float(self.g.edges[etype].data['weight'].float().mean().item())
            else:
                rel_weight = 1.0
            for l in range(layers_per_rel):
                idx = l * num_relations + i
                omega_hat = self._omega_effs[idx]
                loss += rel_weight * (omega_hat - omega_bar) ** 2
        return loss / (num_relations * layers_per_rel)

    def set_focal_alpha_from_labels(self, labels, train_mask, min_alpha=0.5, max_alpha=0.97):
        """Set positive-class focal alpha from the training split imbalance."""
        with torch.no_grad():
            train_labels = labels[train_mask]
            pos = (train_labels == 1).sum().float()
            neg = (train_labels == 0).sum().float()
            total = pos + neg
            if total > 0 and pos > 0:
                alpha = neg / total
                alpha = torch.clamp(alpha, min=min_alpha, max=max_alpha)
                self.focal_alpha = float(alpha.item())
        return self.focal_alpha

    def focal_loss(self, logits, targets):
        """Focal Loss for class-imbalanced bot detection. Eq.(14)
        L_focal = -(1/|V_train|) * Σ_i α_i (1-ŷ_i)^γ log(ŷ_i)
        where α_i = α for class 1 (bots), α_i = 1-α for class 0 (genuine)
        """
        probs = F.softmax(logits, dim=1)
        pt = torch.where(targets == 1, probs[:, 1], probs[:, 0])
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1.0 - self.focal_alpha)
        loss = -alpha_t * (1 - pt) ** self.focal_gamma * torch.log(pt + 1e-8)
        return loss.mean()


class HeteroEnhancedBernsteinNet(EnhancedBernsteinNet):
    """
    TODO: Implement heterogeneous-specific logic or remove if EnhancedBernsteinNet handles all cases.
    Currently identical to EnhancedBernsteinNet, used as placeholder for future heterograph-specific features.
    """
    pass
