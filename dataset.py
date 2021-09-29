
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
#import graph

def sample_or_pad(arr, max_size, pad_value = -1):
        arr_shape = arr.shape

        if arr.size == 0:

            if isinstance(pad_value, list):
                result = np.ones((max_size, len(pad_value)), dtype = arr.dtype) * pad_value
            
            else:

                result = np.ones((max_size,), dtype = arr.dtype) * pad_value

        elif arr.shape[0]>max_size:

            if arr.ndim == 1:
                result = np.random.choice(arr, size = max_size, replace = False)
            else:
                
                idx = np.arange(arr.shape[0])
                np.random.shuffle(idx)
                result = arr[idx[:max_size], :]
        else:
            padding = np.ones((max_size - arr.shape[0],) + arr_shape[1:], dtype = arr.dtype)

            if isinstance(pad_value, list):
                for i in range(len(pad_value)):
                    padding[..., i] *= pad_value[i]
            else:
                padding *= pad_value
            result = np.concatenate((arr, padding), axis = 0)

        return result 
def get_graph_nbrhd_with_rels(train_graph, ent, exclude_tuple):
    """Helper to get neighbor (rels, ents) excluding a particular tuple."""
    es, er, et = exclude_tuple
    neighborhood = [[r, nbr] for nbr in train_graph.kg_data[ent]
                    for r in train_graph.kg_data[ent][nbr]
                    # if r != er]
                    if ent != es or nbr != et or r != er]
    if not neighborhood:
        neighborhood = [[]]
    # if train_graph.add_reverse_graph:
    #   rev_nighborhood = [nbr for nbr in train_graph.reverse_kg_data[ent]
    #                      if ent != et or nbr != es or
    #                      # er not in train_graph.reverse_kg_data[ent][nbr]]
    #                      (train_graph.reverse_kg_data[ent][nbr] - set([er]))]
    #   neighborhood += rev_nighborhood
    neighborhood = np.array(neighborhood, dtype=np.int)
    return neighborhood

def get_graph_nbrhd(train_graph, ent, exclude_tuple):
        es, er, et = exclude_tuple

        neighbourhood = [nbr for nbr in train_graph.kg_data[ent]
                        if ent != es or nbr != et or 
                        er not in train_graph.kg_data[ent][nbr]]
                        #(train_graph.kg_data[ent][nbr] - set([er]))]
        
        neighbourhood = np.array(list(set(neighbourhood)), dtype = np.int)
        
        return neighbourhood


class MyDataset(data.Dataset):

    def __init__(self, data_graph, train_graph=None, mode="train",
                max_negatives=10, max_neighbours=10, num_epochs=20,
                batchsize=64, model_type="attention", val_graph=None, add_inverse = False):

        if not train_graph:
            train_graph = data_graph
        
        self.train_graph = train_graph
        self.data_graph = data_graph
        self.mode = mode
        self.augmented_tuple_store = None
        self.add_inverse = add_inverse


        if mode != "train":

            if max_negatives:
                self.max_negatives = max_negatives
            else:
                self.max_negatives = train_graph.ent_vocab_size-1

        else:
            if not max_negatives and mode == "train":

                raise ValueError("Must provide max_negatives value for training")
            
            self.max_negatives = max_negatives
        
        if max_neighbours:
            self.max_neighbours = max_neighbours
        else:
            self.max_neighbours = train_graph.max_neighbours
        self.input_tensors = None
        self.output_shapes = None
        self.model_type = model_type
        self.val_graph = val_graph

        # if mode == "train":
        #     self.augmented_tuple_store = []
        #     for example in train_graph.tuple_store:
        #         s, r, t = example
        #         self.augmented_tuple_store.append((s,r,t, True))
        #         self.augmented_tuple_store.append((s,r,t, False))
        
        #     self.augmented_tuple_store = np.array(self.augmented_tuple_store)
        
        #print(self.augmented_tuple_store)

        if mode == "train" and self.add_inverse:
            self.augmented_tuple_store = []
            for example in train_graph.tuple_store:
                s, r, t = example
                self.augmented_tuple_store.append((s,r,t, True))
                self.augmented_tuple_store.append((s,r,t, False))
        
            self.augmented_tuple_store = np.array(self.augmented_tuple_store)
        

    
    
    def featurize_example(self, example_tuple):

        s, r, t, reverse = example_tuple

        if not reverse:
            all_targets = self.train_graph.all_reachable_e2[(s, r)]
            if self.mode != "train":
                all_targets |= self.data_graph.all_reachable_e2[(s,r)]
                if self.val_graph:
                    all_targets |= self.val_graph.all_reachable_e2[(s,r)]
        
        else:

            all_targets = self.train_graph.all_reachable_e2_reverse[(t,r)]

            if self.mode != "train":

                all_targets |= self.data_graph.all_reachable_e2_reverse[(t,r)]

                if self.val_graph:
                    all_targets |= self.val_graph.all_reachable_e2_reverse[(t,r)]
            
            s, t = t, s

        candidate_negatives = list(self.train_graph.all_entities - (all_targets | set([t]) | set([self.train_graph.ent_pad])))
        negatives =  sample_or_pad(np.array(candidate_negatives, dtype = np.int), self.max_negatives, pad_value = self.train_graph.ent_pad)


        candidates = np.insert(negatives, 0, t, axis = 0)

        nbrhd_fn = get_graph_nbrhd_with_rels
        #pad_value = self.train_graph.ent_pad
        pad_value = [self.train_graph.rel_pad, self.train_graph.ent_pad]

        nbrs_s = sample_or_pad(nbrhd_fn(self.train_graph, s, (s,r,t)), self.max_neighbours, pad_value = pad_value)
        # nbrs_t = sample_or_pad(nbrhd_fn(self.train_graph, t, (s,r,t)), self.max_neighbours, pad_value = pad_value)
        # nbrs_negatives = np.array([sample_or_pad(nbrhd_fn(self.train_graph, cand, (s,r,t)), 
        #     self.max_neighbours, pad_value = pad_value) for cand in negatives])

        #nbrs_candidates = np.concatenate((np.expand_dims(nbrs_t, 0), nbrs_negatives), axis = 0)
        nbrs_candidates = np.array([], dtype=np.int)


        if self.mode != "train":
            labels = [t]
        
        else:
            
            labels = np.zeros(candidates.shape[0], dtype = np.int)
            labels[0] = 1
            idx = np.arange(candidates.shape[0])
            np.random.shuffle(idx)
            candidates = candidates[idx]

            #nbrs_candidates = nbrs_candidates[idx]
            labels = labels[idx]
        
        return s, nbrs_s, r, candidates, nbrs_candidates, labels


    def __len__(self):
        
        if self.add_inverse:
            return len(self.augmented_tuple_store)
        else:
            return len(self.data_graph.tuple_store)

    
    def __getitem__(self, index):

        if self.add_inverse:

            return self.featurize_example(self.augmented_tuple_store[index])
        else:
            s, r, t = self.data_graph.tuple_store[index]
            return self.featurize_example((s,r,t,True))


    