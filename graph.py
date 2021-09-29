
from collections import defaultdict
import numpy as np
import csv

class Graph(object):

    """
    Read a knowedge graph to memory
    """

    def __init__(
        self, kg_file, entity_vocab = False, relation_vocab = False,
        add_inverse_edge = False, mode = 'train'
    ):


        self.raw_kg_file = kg_file
        self.add_inverse_edge = add_inverse_edge

        if add_inverse_edge:
            self.inverse_relation_prefix = "INVERSE:"

        if entity_vocab:
            self.entity_vocab = entity_vocab
        else:
            self.entity_vocab = {}
        
        if relation_vocab:
            self.relation_vocab = relation_vocab
        else:
            self.relation_vocab = {}

        self.ent_vocab_size = len(self.entity_vocab)
        self.rel_vocab_size = len(self.relation_vocab)
        self._num_edges = 0

        self.kg_data = defaultdict(dict)
        self.next_edges = defaultdict(set)

        self.entity_pad_token = "ePAD"
        self.relation_pad_token = "rPAD"

        self.max_kg_relations = None

        self.mode = mode
        self.read_graph(mode)

        self.inverse_relation_vocab = {v: k for k, v in self.relation_vocab.items()}

        self.inverse_entity_vocab = {v: k for k, v in self.entity_vocab.items()} 

        if mode == "train":

            self.all_entities = set(self.entity_vocab.values())
            self.max_neighbours = self._max_neighbours()

        self.all_reachable_e2  = defaultdict(set)
        self.all_reachable_e2_reverse  = defaultdict(set)
        self.tuple_store = []
        self.create_tuple_store()


    def _max_neighbours(self):

        max_nbrs = 0
        num_nbrs = []
        max_ent = None

        for e1 in self.kg_data:
            nbrs = set(self.kg_data[e1].keys())

            if len(nbrs)> max_nbrs:
                max_nbrs = len(nbrs)
                max_ent = self.inverse_entity_vocab[e1]
            num_nbrs.append(len(nbrs))
        
        #logging.info("Average number of neighbours: %.2f +- %.2f", np.mean(num_nbrs), np.std(num_nbrs))
        #logging.info("Max neighbours %d of entity %s", max_nbrs, max_ent)
        return max_nbrs

    
    def get_inverse_relation_from_name(self, rname):

        if rname.startswith(self.inverse_relation_prefix):
            inv_rname = rname.strip(self.get_inverse_relation_prefix)
        else:
            inv_rname = self.inverse_relation_prefix + rname
        
        return inv_rname

    def get_inverse_relation_from_id(self, r):

        rname = self.inverse_relation_vocab[r]
        
        inv_rname = self.get_inverse_relation_from_name(rname)

        inv_r = self.relation_vocab[inv_rname]

        return inv_r
        
              

    def read_graph(self, mode = "train"):
        """Read the knowledge graph"""

        with open(self.raw_kg_file, "r") as f:
            kg_file = csv.reader(f, delimiter = "\t")
            skipped = 0

            for line in kg_file:

                e1 = line[0].strip()
                if e1 not in self.entity_vocab:
                    if mode != "train":
                        skipped += 1
                        continue
                    self.entity_vocab[e1] = self.ent_vocab_size
                    self.ent_vocab_size+=1
                e1 = self.entity_vocab[e1]

                r = line[1].strip()

                if r not in self.relation_vocab:
                    if mode != "train":
                        skipped +=1
                        continue
                    
                    self.relation_vocab[r] = self.rel_vocab_size
                    self.rel_vocab_size += 1

                if self.add_inverse_edge:
                    inv_r = self.inverse_relation_prefix + r
                    if inv_r not in self.relation_vocab:
                        self.relation_vocab[inv_r] = self.rel_vocab_size
                        self.rel_vocab_size += 1
                    inv_r = self.relation_vocab[inv_r]
                r = self.relation_vocab[r]

                e2 = line[2].strip()
                if e2 not in self.entity_vocab:
                    if mode != "train":
                        skipped +=1
                        continue
                    self.entity_vocab[e2] = self.ent_vocab_size
                    self.ent_vocab_size +=1
                e2 = self.entity_vocab[e2]


                if e2 not in self.kg_data[e1]:
                    self.kg_data[e1][e2] = []
                self.kg_data[e1][e2].append(r)
                self.next_edges[e1].add((r, e2))
                
                if self.add_inverse_edge:
                    if e1 not in self.kg_data[e2]:
                        self.kg_data[e2][e1] = []
                    self.kg_data[e2][e1].append(inv_r)
                    self.next_edges[e2].add((inv_r, e1))
                    self._num_edges += 1

                
                self._num_edges += 1

            if mode == "train" and self.entity_pad_token not in self.entity_vocab:
                self.entity_vocab[self.entity_pad_token] = self.ent_vocab_size
                self.ent_vocab_size += 1

            if mode == "train" and self.relation_pad_token not in self.relation_vocab:
                self.relation_vocab[self.relation_pad_token] = self.rel_vocab_size
                self.rel_vocab_size += 1

            self.ent_pad = self.entity_vocab[self.entity_pad_token]
            self.rel_pad = self.relation_vocab[self.relation_pad_token]


    def create_tuple_store(self, train_graph = None, only_one_hop = False):

        self.tuple_store = []

        skipped = 0

        for e1 in self.kg_data:
            for e2 in self.kg_data[e1]:
                if only_one_hop and train_graph:
                    reachable = e1 in train_graph.kg_data and \
                                e2 in train_graph.kg_data[e1]
                    reachable = reachable or (
                        e1 in train_graph.kg_text_data and \
                        e2 in train_graph.kg_text_data[e1]
                    )
                else:
                    reachable = True

                if reachable:
                    for r in self.kg_data[e1][e2]:
                        self.tuple_store.append((e1, r, e2))

                        self.all_reachable_e2[(e1, r)].add(e2)

                        self.all_reachable_e2_reverse[(e2, r)].add(e1)
                else:
                    skipped += len(self.kg_data[e1][e2])
        self.tuple_store = np.array(self.tuple_store)
