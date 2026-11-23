#!/bin/bash
# Command-line arguments for the benchmark runs.

TFINANCE_SOTA_ARGS=(
    --dataset tfinance
    --hid_dim 32
    --epoch 500
    --homo 1
    --run 1
    --patience 80
    --lambda_freq 0.2
    --lr 0.005
    --reuse_ppr_graph
    --add_edges
    --ppr_topk 32
    --pretrain_seed 0
)

TSOCIAL_SOTA_ARGS=(
    --dataset tsocial
    --hid_dim 10
    --epoch 800
    --homo 1
    --run 1
    --patience 120
    --lambda_freq 0.2
    --lr 0.005
    --reuse_ppr_graph
    --add_edges
    --ppr_topk 64
    --pretrain_seed 0
)

TWIBOT22_SOTA_ARGS=(
    --dataset Twibot-22
    --hid_dim 64
    --epoch 500
    --homo 1
    --run 1
    --patience 80
    --lr 0.003
    --lambda_freq 0.1
    --fp16
    --edge_types _E0,_E1
    --K 3
    --num_scales 5
)

TWIBOT20_SOTA_ARGS=(
    --dataset Twibot-20
    --hid_dim 192
    --epoch 500
    --homo 1
    --run 1
    --patience 100
    --lr 0.005
    --lambda_freq 0.2
    --fp16
    --split_mode stratified
    --seed 1
)

# The single-model benchmark uses a stratified split.
MGTAB_STABLE_STRATIFIED_ARGS=(
    --dataset MGTAB
    --epoch 800
    --homo 1
    --run 1
    --patience 180
    --split_mode stratified
    --split_seed 1
    --seed 1
    --mgtab_feature_mode concat
    --mgtab_feature_norm
    --mgtab_feature_seed 2
    --mgtab_embedding_scale 1.55
    --hid_dim 112
    --lambda_freq 0.4
    --lr 0.005
    --dropout_rate 0.25
    --final_residual_weight 0.30
    --focal_gamma 1.5
)

MGTAB_BEST_REPORTABLE_STABLE_ARGS=(
    "${MGTAB_STABLE_STRATIFIED_ARGS[@]}"
)

# The MGTAB ensemble uses a random split.
MGTAB_RANDOM_ALPHA078_ARGS=(
    --dataset MGTAB
    --epoch 800
    --homo 1
    --run 1
    --patience 180
    --split_mode random
    --split_seed 2
    --seed 2
    --mgtab_feature_mode concat
    --mgtab_feature_norm
    --mgtab_feature_seed 2
    --mgtab_embedding_scale 1.55
    --hid_dim 112
    --lambda_freq 0.4
    --lr 0.005
    --dropout_rate 0.25
    --final_residual_weight 0.30
    --focal_gamma 1.5
    --no_auto_focal_alpha
    --focal_alpha 0.78
)

MGTAB_BEST_RANDOM_NEAR_PAPER_ARGS=(
    "${MGTAB_RANDOM_ALPHA078_ARGS[@]}"
)

MGTAB_BEST_RANDOM_SINGLE_PEAK_ARGS=(
    "${MGTAB_RANDOM_ALPHA078_ARGS[@]}"
)
