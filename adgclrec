import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
import os
import random
from collections import deque, defaultdict
from tensorflow.keras import layers, Model
import time
import itertools
import sys
import tensorflow_probability as tfp

# File Logger
class FileLogger:

    def __init__(self, filename="model_log.txt"):
        self.terminal = sys.stdout
        with open(filename, "w") as f: pass
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self, *args, **kwargs):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if self.log: self.log.close()

# Reproducibility and Environment Setup
# os.environ['PYTHONHASHSEED'] = '42'
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

np.set_printoptions(precision=6, suppress=True)

# Data Loading
def load_interactions(file_path, sep_pattern=r"\s+", header=None,
                      names=['user','item','weight'], has_prefix=True):

    df = pd.read_csv(
        file_path, sep=sep_pattern, engine='python', header=header,
        names=names, dtype=str
    )
    if has_prefix:
        df['user'] = df['user'].str.lstrip('u').astype(int)
        df['item'] = df['item'].str.lstrip('i').astype(int)
    else:
        df['user'] = df['user'].astype(int)
        df['item'] = df['item'].astype(int)
    return df

def build_mappings_from_train(train_df):

    users = sorted(train_df['user'].unique())
    items = sorted(train_df['item'].unique())
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}
    return user_map, item_map

# Graph Utilities
def build_joint_adjacency_and_normalize(train_df, user_map, item_map, num_users, num_items):

    df_filtered = train_df[train_df['user'].isin(user_map) & train_df['item'].isin(item_map)]
    
    rows_r = df_filtered['user'].map(user_map).values
    cols_r = df_filtered['item'].map(item_map).values
    data_r = np.ones(len(df_filtered), dtype=np.float32)

    # Construct the full adjacency matrix A_hat
    # The shape of the full matrix is (num_users + num_items)*(num_users + num_items)

    adj_rows_r = rows_r
    adj_cols_r = cols_r + num_users
    
    adj_rows_rt = cols_r + num_users
    adj_cols_rt = rows_r

    all_rows = np.concatenate([adj_rows_r, adj_rows_rt])
    all_cols = np.concatenate([adj_cols_r, adj_cols_rt])
    all_data = np.concatenate([data_r, data_r])

    A_hat_coo = coo_matrix((all_data, (all_rows, all_cols)),
                           shape=(num_users + num_items, num_items + num_users))
                           
    degrees = np.array(A_hat_coo.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1e-6 # Small epsilon for numerical stability
    
    D_inv_sqrt = sparse.diags(np.power(degrees, -0.5))

    normalized_adj_coo = D_inv_sqrt.dot(A_hat_coo).dot(D_inv_sqrt).tocoo()

    indices = tf.stack([tf.convert_to_tensor(normalized_adj_coo.row, dtype=tf.int64),
                        tf.convert_to_tensor(normalized_adj_coo.col, dtype=tf.int64)], axis=1)
    values = tf.convert_to_tensor(normalized_adj_coo.data, dtype=tf.float32)
    shape = tf.constant(normalized_adj_coo.shape, dtype=tf.int64)

    return tf.SparseTensor(indices, values, shape)


# Core Contrastive and Model Components
class CrossViewNTXentLoss(layers.Layer):

    def __init__(self, temperature=0.2, **kwargs):

        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, list_of_noisy_views):

        V = len(list_of_noisy_views)
        if V != 2:
            return 0.0 

        batch_size = tf.shape(list_of_noisy_views[0])[0]

        z_v0 = tf.math.l2_normalize(list_of_noisy_views[0], axis=1)
        z_v1 = tf.math.l2_normalize(list_of_noisy_views[1], axis=1)

        representations = tf.concat([z_v0, z_v1], axis=0)
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True)

        # Positives
        l_pos = tf.linalg.diag_part(similarity_matrix, k=batch_size)
        r_pos = tf.linalg.diag_part(similarity_matrix, k=-batch_size)
        positives = tf.concat([l_pos, r_pos], axis=0)

        # Negatives
        mask = tf.eye(2 * batch_size, dtype=tf.bool) # Diagonal mask for self-similarity
        negatives = tf.reshape(tf.boolean_mask(similarity_matrix, tf.logical_not(mask)), [2 * batch_size, -1])

        logits = tf.concat([tf.expand_dims(positives, 1), negatives], axis=1) / self.temperature
        labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)

        total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
        
        return total_loss

# Noise Generator
class EpsilonGenerator(Model):

    def __init__(self, num_layers, embed_dim=128, **kwargs):

        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layer_embedding = layers.Embedding(num_layers, embed_dim, name="gen_layer_emb") # Learnable embedding for layer_id
        self.dense1 = layers.Dense(embed_dim, activation='relu', name="gen_dense1")
        self.mean_layer = layers.Dense(1, name="gen_mean")
        self.log_std_layer = layers.Dense(1, name="gen_log_std", bias_initializer=tf.keras.initializers.Constant(-2.0))

    def call(self, layer_id):

        l_emb = self.layer_embedding(layer_id)

        if l_emb.shape.rank == 1:
            l_emb_expanded = tf.expand_dims(l_emb, 0)
        else:
            l_emb_expanded = l_emb
        
        hidden_output = self.dense1(l_emb_expanded)
        
        mean = self.mean_layer(hidden_output)
        log_std = self.log_std_layer(hidden_output)

        return tf.squeeze(mean, axis=-1), tf.squeeze(log_std, axis=-1)


class LightGCNModel(Model):

    def __init__(self, num_users, num_items, embed_dim, num_layers, epsilon_generator):

        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_noisy_views = 2
        self.epsilon_generator = epsilon_generator
        self.user_embed = layers.Embedding(num_users, embed_dim, name="user_embedding", embeddings_initializer='random_normal')
        self.item_embed = layers.Embedding(num_items, embed_dim, name="item_embedding", embeddings_initializer='random_normal')
        self.scorer_dense1 = layers.Dense(64, activation='relu', name="scorer_dense1")
        self.scorer_dense2 = layers.Dense(1, name="scorer_dense2")
        self.generated_epsilons = None
        self.generated_epsilon_means = None
        self.generated_epsilon_log_stds = None

    def call(self, adj_tf, training=False):

        ego_embeddings = tf.concat([self.user_embed(tf.range(self.num_users)),
                                    self.item_embed(tf.range(self.num_items))], axis=0)

        all_z_clean_online = []
        all_z_noisy_online_per_view = [[] for _ in range(self.num_noisy_views)]
        sampled_epsilons_ta = tf.TensorArray(tf.float32, size=self.num_layers * self.num_noisy_views, dynamic_size=False, clear_after_read=False)
        epsilon_means_ta = tf.TensorArray(tf.float32, size=self.num_layers * self.num_noisy_views, dynamic_size=False, clear_after_read=False)
        epsilon_log_stds_ta = tf.TensorArray(tf.float32, size=self.num_layers * self.num_noisy_views, dynamic_size=False, clear_after_read=False)
        eps_idx = 0

        current_online_clean_view = None
        current_online_noisy_view_0 = None
        current_online_noisy_view_1 = None

        # First Layer (L=0)
        h_l0_propagated_clean = tf.sparse.sparse_dense_matmul(adj_tf, ego_embeddings)
        all_z_clean_online.append(h_l0_propagated_clean)
        current_online_clean_view = h_l0_propagated_clean

        mean_l0, log_std_l0 = self.epsilon_generator(tf.constant(0, dtype=tf.int32))
        
        if training:
            std_l0 = tf.exp(log_std_l0) + 1e-6
            epsilon_dist_l0 = tfp.distributions.Normal(loc=mean_l0, scale=std_l0)

            epsilon_l0_v0_sampled_raw = epsilon_dist_l0.sample()
            epsilon_l0_v0 = tf.nn.softplus(epsilon_l0_v0_sampled_raw)
            
            epsilon_l0_v1_sampled_raw = epsilon_dist_l0.sample()
            epsilon_l0_v1 = tf.nn.softplus(epsilon_l0_v1_sampled_raw)
        else:
            epsilon_l0_v0 = tf.nn.softplus(mean_l0)
            epsilon_l0_v1 = tf.nn.softplus(mean_l0) # Both views use mean in eval mode

        sampled_epsilons_ta = sampled_epsilons_ta.write(eps_idx, epsilon_l0_v0)
        epsilon_means_ta = epsilon_means_ta.write(eps_idx, mean_l0)
        epsilon_log_stds_ta = epsilon_log_stds_ta.write(eps_idx, log_std_l0)
        eps_idx += 1
        z_noisy_online_v0_l0 = self.add_sim_gcl_noise(h_l0_propagated_clean, epsilon_l0_v0, training)
        all_z_noisy_online_per_view[0].append(z_noisy_online_v0_l0)
        current_online_noisy_view_0 = z_noisy_online_v0_l0

        sampled_epsilons_ta = sampled_epsilons_ta.write(eps_idx, epsilon_l0_v1)
        epsilon_means_ta = epsilon_means_ta.write(eps_idx, mean_l0)
        epsilon_log_stds_ta = epsilon_log_stds_ta.write(eps_idx, log_std_l0)
        eps_idx += 1
        z_noisy_online_v1_l0 = self.add_sim_gcl_noise(h_l0_propagated_clean, epsilon_l0_v1, training)
        all_z_noisy_online_per_view[1].append(z_noisy_online_v1_l0)
        current_online_noisy_view_1 = z_noisy_online_v1_l0


        # Subsequent LightGCN Layers (L=1 to num_layers-1)
        for l in range(1, self.num_layers):

            propagated_online_clean = tf.sparse.sparse_dense_matmul(adj_tf, current_online_clean_view)
            all_z_clean_online.append(propagated_online_clean)
            current_online_clean_view = propagated_online_clean

            mean_l, log_std_l = self.epsilon_generator(tf.constant(l, dtype=tf.int32))
            
            if training:
                std_l = tf.exp(log_std_l) + 1e-6
                epsilon_dist_l = tfp.distributions.Normal(loc=mean_l, scale=std_l)

                epsilon_lv0_sampled_raw = epsilon_dist_l.sample()
                epsilon_lv0 = tf.nn.softplus(epsilon_lv0_sampled_raw)

                epsilon_lv1_sampled_raw = epsilon_dist_l.sample()
                epsilon_lv1 = tf.nn.softplus(epsilon_lv1_sampled_raw)
            else:
                epsilon_lv0 = tf.nn.softplus(mean_l)
                epsilon_lv1 = tf.nn.softplus(mean_l) # Both views use mean in eval mode

            sampled_epsilons_ta = sampled_epsilons_ta.write(eps_idx, epsilon_lv0)
            epsilon_means_ta = epsilon_means_ta.write(eps_idx, mean_l)
            epsilon_log_stds_ta = epsilon_log_stds_ta.write(eps_idx, log_std_l)
            eps_idx += 1
            propagated_online_noisy = tf.sparse.sparse_dense_matmul(adj_tf, current_online_noisy_view_0)
            z_noisy_online_lv0 = self.add_sim_gcl_noise(propagated_online_noisy, epsilon_lv0, training)
            all_z_noisy_online_per_view[0].append(z_noisy_online_lv0)
            current_online_noisy_view_0 = z_noisy_online_lv0

            sampled_epsilons_ta = sampled_epsilons_ta.write(eps_idx, epsilon_lv1)
            epsilon_means_ta = epsilon_means_ta.write(eps_idx, mean_l)
            epsilon_log_stds_ta = epsilon_log_stds_ta.write(eps_idx, log_std_l)
            eps_idx += 1
            propagated_online_noisy = tf.sparse.sparse_dense_matmul(adj_tf, current_online_noisy_view_1)
            z_noisy_online_lv1 = self.add_sim_gcl_noise(propagated_online_noisy, epsilon_lv1, training)
            all_z_noisy_online_per_view[1].append(z_noisy_online_lv1)
            current_online_noisy_view_1 = z_noisy_online_lv1

        self.generated_epsilons = tf.reshape(sampled_epsilons_ta.stack(), (self.num_layers, self.num_noisy_views))
        self.generated_epsilon_means = tf.reshape(epsilon_means_ta.stack(), (self.num_layers, self.num_noisy_views))
        self.generated_epsilon_log_stds = tf.reshape(epsilon_log_stds_ta.stack(), (self.num_layers, self.num_noisy_views))

        final_z_clean_online = tf.reduce_mean(tf.stack(all_z_clean_online, axis=0), axis=0)
        final_z_noisy_online_views = [tf.reduce_mean(tf.stack(view_list, axis=0), axis=0)
                                      for view_list in all_z_noisy_online_per_view]

        final_user_embs_clean_online, final_item_embs_clean_online = tf.split(final_z_clean_online, [self.num_users, self.num_items], 0)
        
        final_user_embs_noisy_online = []
        final_item_embs_noisy_online = []
        for v_idx in range(self.num_noisy_views):
            u_noisy, i_noisy = tf.split(final_z_noisy_online_views[v_idx], [self.num_users, self.num_items], 0)
            final_user_embs_noisy_online.append(u_noisy)
            final_item_embs_noisy_online.append(i_noisy)

        return (final_user_embs_clean_online, final_item_embs_clean_online,
                final_user_embs_noisy_online, final_item_embs_noisy_online)

    def add_sim_gcl_noise(self, embeddings, epsilon, training):

        if not training:
            return embeddings

        random_noise = tf.random.uniform(tf.shape(embeddings))
        noise_magnitude = tf.nn.l2_normalize(random_noise, axis=1) * tf.cast(epsilon, tf.float32)
        return embeddings + tf.multiply(tf.sign(embeddings), noise_magnitude)


# Evaluation
def evaluate_model(gnn_model, train_df, user_map, item_map, test_adj_coo, reverse_user_map, reverse_item_map, num_users, num_items):

    print("\n--- Starting Evaluation ---")
    k_values=[1, 3, 5, 10, 20]
    
    joint_adj_tf = build_joint_adjacency_and_normalize(train_df, user_map, item_map, num_users, num_items)
    user_embs_clean_online, item_embs_clean_online, _, _ = gnn_model(joint_adj_tf, training=False)

    user_feats = user_embs_clean_online
    item_feats = item_embs_clean_online

    test_users = np.unique(test_adj_coo.row)
    all_metrics = {k: defaultdict(list) for k in k_values}
    per_user_metrics = {}
    train_adj_csr, test_adj_csr = train_adj_coo.tocsr(), test_adj_coo.tocsr()
    max_k = max(k_values)

    for u_idx in test_users:
        u_emb = tf.expand_dims(user_feats[u_idx], 0)
        
        u_emb_expanded = tf.repeat(u_emb, repeats=num_items, axis=0)
        item_embs_repeated = tf.tile(item_feats, [1, 1])
        concatenated_pairs = tf.concat([u_emb_expanded, item_embs_repeated], axis=1)
        all_scores = tf.squeeze(gnn_model.scorer_dense2(gnn_model.scorer_dense1(concatenated_pairs)), axis=-1)

        train_items_for_user = train_adj_csr[u_idx].indices

        # Mask out already interacted items from recommendations
        all_scores = tf.tensor_scatter_nd_update(all_scores, tf.expand_dims(train_items_for_user, 1), tf.ones(len(train_items_for_user)) * -np.inf)
        
        true_items = test_adj_csr[u_idx].indices
        if len(true_items) == 0: continue
            
        _, top_k_indices = tf.math.top_k(all_scores, k=max_k)
        top_k_indices = top_k_indices.numpy()
        is_hit = np.isin(top_k_indices, true_items)

        for k in k_values:
            is_hit_k = is_hit[:k]
            num_hits = is_hit_k.sum()
            
            precision_k = num_hits / k
            recall_k = num_hits / len(true_items)
            
            all_metrics[k]['precision'].append(precision_k)
            all_metrics[k]['recall'].append(recall_k)
            
            dcg = np.sum(is_hit_k / np.log2(np.arange(2, k + 2)))
            idcg = np.sum(1./np.log2(np.arange(2, len(true_items)+ 2)))
            all_metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)
            
            hits_at_k = np.where(is_hit_k)[0]
            if len(hits_at_k) > 0:
                sum_precisions = np.sum((np.arange(len(hits_at_k)) + 1) / (hits_at_k + 1))
                avg_precision = sum_precisions / len(true_items)
                reciprocal_rank = 1.0 / (hits_at_k[0] + 1)
            else:
                avg_precision = 0.0
                reciprocal_rank = 0.0
            all_metrics[k]['map'].append(avg_precision)
            all_metrics[k]['mrr'].append(reciprocal_rank)

            if k == 10:
                f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
                top_10_recs_indices = top_k_indices[:10]
                top_10_recs_original_ids = [reverse_item_map.get(i, -1) for i in top_10_recs_indices]
                original_user_id = reverse_user_map.get(u_idx, -1)
                per_user_metrics[original_user_id] = {
                    'f1_at_10': f1_k,
                    'top_10_recs': ','.join(map(str, top_10_recs_original_ids))
                }

    final_results = {}
    print("\n--- Evaluation Results ---")
    for k in sorted(all_metrics.keys()):
        mean_precision = np.mean(all_metrics[k]['precision'])
        mean_recall = np.mean(all_metrics[k]['recall'])
        mean_ndcg = np.mean(all_metrics[k]['ndcg'])
        mean_map = np.mean(all_metrics[k]['map'])
        mean_mrr = np.mean(all_metrics[k]['mrr'])
        f1 = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
        print(f"k={k}: F1={f1:.6f} Precision={mean_precision:.6f} Recall={mean_recall:.6f} NDCG={mean_ndcg:.6f} MAP={mean_map:.6f} MRR={mean_mrr:.6f}")
        if k == 10:
            final_results = {'f1': f1, 'ndcg': mean_ndcg, 'map': mean_map, 'mrr': mean_mrr}
            
    return final_results, per_user_metrics


def train_model(joint_adj_tf, train_adj_coo, test_adj_coo, gnn_model, epsilon_generator, hparams, reverse_user_map, reverse_item_map, num_users, num_items, train_df, user_map, item_map):

    # Two optimizers: one for the main model (GNN) and one for the Epsilon Generator
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['lr'], epsilon=1e-8) 
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['gen_lr'], epsilon=1e-8)
     
    cross_view_nt_xent_loss_fn = CrossViewNTXentLoss(temperature=hparams['cl_temperature'])

    train_adj_csr = train_adj_coo.tocsr()
    user_pos_items = {u: train_adj_csr[u].indices for u in range(num_users)}
    
    users_all, items_all = train_adj_coo.nonzero()
    
    best_f1_so_far = 0.0

    @tf.function(autograph=False)
    def train_step(u_batch_indices, v_batch_indices, v_neg_rank_indices, batch_idx, current_lambda_cl):
        # Generator Training Step
        with tf.GradientTape() as gen_tape:

            user_embs_clean_online_full, item_embs_clean_online_full, \
            user_embs_noisy_online_views_full, item_embs_noisy_online_views_full = gnn_model(joint_adj_tf, training=True)

            all_batch_user_ids = u_batch_indices
            all_batch_item_ids = tf.concat([v_batch_indices, v_neg_rank_indices], axis=0)

            batch_noisy_online_views = []
            for v_idx in range(gnn_model.num_noisy_views): # Iterate for the two noisy views
                batch_user_noisy_online = tf.gather(user_embs_noisy_online_views_full[v_idx], all_batch_user_ids)
                batch_item_noisy_online = tf.gather(item_embs_noisy_online_views_full[v_idx], all_batch_item_ids)
                batch_noisy_online_views.append(tf.concat([batch_user_noisy_online, batch_item_noisy_online], axis=0))

            # Gen CL
            L_cl = cross_view_nt_xent_loss_fn(batch_noisy_online_views)
            
            # Gen KL
            prior_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
            learned_dist = tfp.distributions.Normal(loc=gnn_model.generated_epsilon_means[:, 0], 
                                                    scale=tf.exp(gnn_model.generated_epsilon_log_stds[:, 0]))
            L_kl_gen = tf.reduce_mean(tfp.distributions.kl_divergence(learned_dist, prior_dist))

            # Generator Loss
            gen_loss = -L_cl + hparams['gamma_kl'] * L_kl_gen

        gen_trainable_vars = epsilon_generator.trainable_variables
        gen_grads = gen_tape.gradient(gen_loss, gen_trainable_vars)
        generator_optimizer.apply_gradients(zip(gen_grads, gen_trainable_vars))

        # Main Model Training Step
        # May generate build warning during training but not important because the optimizer sees that 
        # it can't update epsilons while it is visible to it. 
        # We update epsilons with the generator optimizer.
        with tf.GradientTape() as model_tape:
            
            user_embs_clean_online_full, item_embs_clean_online_full, \
            user_embs_noisy_online_views_full, item_embs_noisy_online_views_full = gnn_model(joint_adj_tf, training=True)

            all_batch_user_ids = u_batch_indices
            all_batch_item_ids = tf.concat([v_batch_indices, v_neg_rank_indices], axis=0)

            batch_user_clean_online = tf.gather(user_embs_clean_online_full, all_batch_user_ids)
            batch_item_clean_online = tf.gather(item_embs_clean_online_full, all_batch_item_ids)

            batch_noisy_online_views = []
            for v_idx in range(gnn_model.num_noisy_views):
                batch_user_noisy_online = tf.gather(user_embs_noisy_online_views_full[v_idx], all_batch_user_ids)
                batch_item_noisy_online = tf.gather(item_embs_noisy_online_views_full[v_idx], all_batch_item_ids)
                batch_noisy_online_views.append(tf.concat([batch_user_noisy_online, batch_item_noisy_online], axis=0))

            u_batch_agg_embs = tf.gather(batch_user_clean_online, tf.range(tf.shape(u_batch_indices)[0])) 
            v_batch_agg_embs = tf.gather(batch_item_clean_online, tf.range(tf.shape(v_batch_indices)[0])) 
            v_neg_rank_agg_embs = tf.gather(batch_item_clean_online, tf.range(tf.shape(v_batch_indices)[0], tf.shape(all_batch_item_ids)[0])) 

            pos_pairs_concatenated = tf.concat([u_batch_agg_embs, v_batch_agg_embs], axis=1)
            neg_pairs_concatenated = tf.concat([u_batch_agg_embs, v_neg_rank_agg_embs], axis=1)
            
            phi_p = tf.squeeze(gnn_model.scorer_dense2(gnn_model.scorer_dense1(pos_pairs_concatenated)), axis=-1)
            phi_n = tf.squeeze(gnn_model.scorer_dense2(gnn_model.scorer_dense1(neg_pairs_concatenated)), axis=-1)
            
            # Main Rank
            L_rank = -tf.reduce_mean(tf.math.log_sigmoid(phi_p - phi_n))

            reg_loss = hparams['reg_lambda'] * (
                tf.nn.l2_loss(u_batch_agg_embs) + 
                tf.nn.l2_loss(v_batch_agg_embs) + 
                tf.nn.l2_loss(v_neg_rank_agg_embs)
            ) / tf.cast(tf.shape(u_batch_agg_embs)[0], tf.float32)
            L_rank += reg_loss

            # Main CL
            L_cl_model_loss = cross_view_nt_xent_loss_fn(batch_noisy_online_views)

            total_loss = (tf.cast(hparams['w_rank'] * L_rank, tf.float32) +
                          tf.cast(current_lambda_cl * L_cl_model_loss, tf.float32))

        model_trainable_vars = gnn_model.trainable_weights
        model_grads = model_tape.gradient(total_loss, model_trainable_vars)
        model_optimizer.apply_gradients(zip(model_grads, model_trainable_vars))
        
        return total_loss, L_rank, L_cl_model_loss, L_cl, L_kl_gen

    # Main Epoch Loop
    for epoch in range(1, hparams['epochs'] + 1):
        print(f"\nEpoch {epoch}/{hparams['epochs']}")
        epoch_start_time = time.time()

        # Change the if condition to dynamically determine current_lambda_cl.
        # if you want to receive turn-based updates from cl loss for main model.
        # Leave the right side at 0 to get updates for both every epoch.
        if (epoch - 1) % 10 < 0:
            current_lambda_cl_for_model = 0.0
        else:
            current_lambda_cl_for_model = hparams['lambda_cl']
            print(f" lambda_cl = {current_lambda_cl_for_model}")

        current_epoch_negatives = pregenerate_negatives(num_users, num_items, user_pos_items, num_neg_samples_per_user=300)
        shuffled_indices = np.random.permutation(len(users_all))
        epoch_losses = defaultdict(float)
        num_batches = len(users_all) // hparams['batch_size']

        for i in range(num_batches):
            batch_indices = shuffled_indices[i*hparams['batch_size']:(i+1)*hparams['batch_size']]
            u_batch_indices_py = users_all[batch_indices]
            v_batch_indices_py = items_all[batch_indices]
            
            v_neg_rank_indices_py = [random.choice(current_epoch_negatives[u]) for u in u_batch_indices_py]
            
            u_batch_indices = tf.convert_to_tensor(u_batch_indices_py, dtype=tf.int32)
            v_batch_indices = tf.convert_to_tensor(v_batch_indices_py, dtype=tf.int32)
            v_neg_rank_indices = tf.convert_to_tensor(v_neg_rank_indices_py, dtype=tf.int32)
            
            total_loss, L_rank, L_cl_model_loss, L_cl, L_kl_gen = train_step(
                u_batch_indices, v_batch_indices, v_neg_rank_indices, tf.constant(i, dtype=tf.int32),
                current_lambda_cl_for_model
            )

            epoch_losses['total'] += total_loss
            epoch_losses['rank'] += L_rank
            epoch_losses['cl_gen'] += L_cl
            epoch_losses['cl_model'] += L_cl_model_loss
            epoch_losses['kl_gen'] += L_kl_gen

        for loss_name, loss_val in epoch_losses.items():
            print(f"  Avg {loss_name} Loss: {loss_val.numpy() / num_batches:.6f}")
        
        # Print current epsilon values (final state after epoch)
        # Call in eval mode to avoid noise
        # _, _, _, _ = gnn_model(joint_adj_tf, training=False) 
        # current_eps_values = gnn_model.generated_epsilons.numpy()
        # current_eps_means = gnn_model.generated_epsilon_means.numpy()
        # current_eps_log_stds = gnn_model.generated_epsilon_log_stds.numpy()
        # print(f"  Final Sampled Epsilon Values (Layer x Noisy View) for Epoch {epoch}:\n{current_eps_values}")
        # print(f"  Final Epsilon Means (Layer x Noisy View) for Epoch {epoch}:\n{current_eps_means}")
        # print(f"  Final Epsilon Log_Stds (Layer x Noisy View) for Epoch {epoch}:\n{current_eps_log_stds}")

        epoch_end_time = time.time()
        print(f"Epoch Duration: {(epoch_end_time - epoch_start_time):.2f} s")
        
        eval_start_time = time.time()
        dev_results, per_user_metrics = evaluate_model(
            gnn_model, train_df, user_map, item_map, test_adj_coo,
            reverse_user_map, reverse_item_map, num_users, num_items
        )
        eval_end_time = time.time()
        print(f"Evaluation Duration: {(eval_end_time - eval_start_time):.2f} s")

        current_f1 = dev_results.get('f1', 0.0)
        if current_f1 > best_f1_so_far:
            print(f"New best F1@10 score: {current_f1:.6f}. Saving model weights.")
            best_f1_so_far = current_f1
            gnn_model.save_weights("best_main_model_weights.weights.h5")
            epsilon_generator.save_weights("best_epsilon_generator.weights.h5")
            
            f1_data_list = []
            for user_id, metrics in per_user_metrics.items():
                f1_data_list.append({
                    'original_user_id': user_id,
                    'f1_at_10': metrics['f1_at_10'],
                    'top_10_recs': metrics['top_10_recs']
                })
            
            f1_df = pd.DataFrame(f1_data_list)
            f1_df.to_csv(f"best_f1_scores_per_user_epoch_{epoch}_simplified.csv", index=False)
            print(f"Saved per-user metrics to best_f1_scores_per_user_epoch_{epoch}_simplified.csv")


        print(f"Best f1 so far: {best_f1_so_far:.6f}")

# Pregenerate a pool of random negatives for each user for each epoch to sample from.
# Significantly helps with speed and avoids tons of python overhead.
def pregenerate_negatives(num_users, num_items, user_pos_items, num_neg_samples_per_user=50):
    all_negatives = {}
    for u_idx in range(num_users):
        positive_items = user_pos_items.get(u_idx, set())
        negatives_for_user = []
        while len(negatives_for_user) < num_neg_samples_per_user:
            rand_item = random.randrange(num_items)
            if rand_item not in positive_items:
                negatives_for_user.append(rand_item)
        all_negatives[u_idx] = negatives_for_user
    return all_negatives

# Main Execution
if __name__ == "__main__":
    sys.stdout = FileLogger("model_log_simplified_pipeline.txt")

    # Hyperparameters
    HPARAMS = {
        'embed_dim': 128, 
        'num_layers': 2,
        'batch_size': 128, 
        'epochs': 150, 
        'lr': 1e-4,
        'gen_lr': 1e-4,
        'cl_temperature': 0.2,
        'w_rank': 1.0,
        'lambda_cl': 0.7,
        'gamma_kl': 0.01,
        'reg_lambda': 0.0001,
        'print_epsilon_freq_batches': 1000
    }

    # Data Loading
    train_df = load_interactions('ml100k_train.txt', has_prefix=False)
    test_df  = load_interactions('ml100k_test.txt', has_prefix=False)
    user_map, item_map = build_mappings_from_train(train_df)
    reverse_user_map = {idx: u for u, idx in user_map.items()}
    reverse_item_map = {idx: i for i, idx in item_map.items()} 

    num_users, num_items = len(user_map), len(item_map)

    train_adj_coo = coo_matrix((np.ones(len(train_df), dtype=np.float32), 
                                                (train_df['user'].map(user_map).values, train_df['item'].map(item_map).values)),shape=(num_users, num_items))

    test_adj_coo = coo_matrix((np.ones(len(test_df), dtype=np.float32), 
                               (test_df['user'].map(user_map).values, test_df['item'].map(item_map).values)), 
                              shape=(num_users, num_items))

    joint_adj_tf = build_joint_adjacency_and_normalize(train_df, user_map, item_map, num_users, num_items)


    print(f"# Users: {num_users}")
    print(f"# Items: {num_items}")

    # Model Initialization
    epsilon_generator = EpsilonGenerator(num_layers=HPARAMS['num_layers'], embed_dim=HPARAMS['embed_dim'])
    gnn_model = LightGCNModel(num_users=num_users, num_items=num_items, 
                              embed_dim=HPARAMS['embed_dim'], num_layers=HPARAMS['num_layers'],
                              epsilon_generator=epsilon_generator)

    train_model(joint_adj_tf, train_adj_coo, test_adj_coo, gnn_model, epsilon_generator, HPARAMS, reverse_user_map, reverse_item_map, num_users, num_items, train_df, user_map, item_map)
