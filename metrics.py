from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import numpy as np
import torch

def mrr_calc(scores_torch, candidates, labels):

    #scores = tf.convert_to_tensor(scores.detach().cpu().numpy())
    scores = tf.convert_to_tensor(scores_torch.clone().detach().cpu().numpy())
    candidates = tf.convert_to_tensor(candidates.cpu().numpy())
    #print(labels[0].numpy())
    labels = tf.convert_to_tensor(labels[0].numpy())
    labels = tf.reshape(labels, (scores.shape[0],1))

    _, top_score_ids = tf.nn.top_k(scores, k=tf.shape(scores)[-1])
    
    batch_indices = tf.cumsum(
        tf.ones_like(candidates, dtype=tf.int32), axis=0, exclusive=True
    )
    indices = tf.concat([tf.expand_dims(batch_indices, axis=-1),
                        tf.expand_dims(top_score_ids, -1)], -1)
    sorted_candidates = tf.gather_nd(candidates, indices)
    # label_ids = tf.expand_dims(tf.argmax(labels, axis=1), 1)
    #print(sorted_candidates.shape)
    #print(labels.shape)
    label_rank_indices = tf.where(
        tf.equal(sorted_candidates, labels)
    )
    # +1 because top rank should be 1 not 0
    ranks = label_rank_indices[:, 1] + 1
    rr = 1.0 / tf.cast(ranks, tf.float32)
    
    rr = torch.from_numpy(rr.numpy())
    #scores = torch.tensor(scores.numpy(), requires_grad = True)
    candidates = torch.tensor(candidates.numpy())
    labels = torch.tensor(labels.numpy())

    return rr  # , ranks, label_rank_indices, sorted_candidates, top_score_ids


def hits_at_k(scores_torch, candidates, labels, k=10):

    #scores = tf.convert_to_tensor(scores.detach().cpu().numpy())
    scores = tf.convert_to_tensor(scores_torch.clone().detach().cpu().numpy())
    candidates = tf.convert_to_tensor(candidates.cpu().numpy())
    labels = tf.convert_to_tensor(labels[0].numpy())
    labels = tf.reshape(labels, (scores.shape[0],1))

    _, top_score_ids = tf.nn.top_k(scores, k=k)
    batch_indices = tf.cumsum(
        tf.ones_like(top_score_ids, dtype=tf.int32), axis=0, exclusive=True
    )
    indices = tf.concat([tf.expand_dims(batch_indices, axis=-1),
                        tf.expand_dims(top_score_ids, -1)], -1)
    sorted_candidates = tf.gather_nd(candidates, indices)
    # label_ids = tf.expand_dims(tf.argmax(labels, axis=1), 1)
    #print(sorted_candidates.shape)
    hits = tf.reduce_max(
        tf.cast(tf.equal(sorted_candidates, labels), tf.float32), 1
    )

    #scores = torch.tensor(scores.numpy(), requires_grad=True)
    candidates = torch.tensor(candidates.numpy())
    labels = torch.tensor(labels.numpy())

    hits = torch.from_numpy(hits.numpy())
    return hits  # , sorted_candidates, top_score_ids