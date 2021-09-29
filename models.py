from torch.utils import data
import torch
from torch.utils import data as torch_data
import torch.optim as optim
import numpy as np
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, train_graph, is_train, train_dropout = 0.5, dim = 100):

        super(Attention, self).__init__()
        self.entity_count = train_graph.ent_vocab_size
        self.relation_count = train_graph.rel_vocab_size
        self.dim = dim
        self.model = {}
        self.is_train = is_train
        self.dropout_p = train_dropout
        self.train_graph = train_graph

        self.entities_emb = self._init_entity_emb()
        self.model['entities_emb'] = self.entities_emb

        self.init_entities_emb = self.entities_emb
        self.model['init_entities_emb'] = self.init_entities_emb

        self.relations_emb = self._init_relation_emb()
        self.model['relations_emb'] = self.entities_emb

        self.attention_emb_n = self._init_attention_embedding()
        self.model['attention_emb_n'] = self.attention_emb_n

        self.attention_emb_s = self._init_attention_embedding()
        self.model['attention_emb_s'] = self.attention_emb_s

        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_p)

        #self.criterion = nn.MarginRankingLoss(margin = margin, reduction='none')

    def _init_entity_emb(self):

        entities_emb = nn.Embedding(num_embeddings = self.entity_count, embedding_dim = self.dim)

        torch.nn.init.xavier_uniform_(entities_emb.weight.data)
            
        return entities_emb

    def _init_relation_emb(self):

        relations_emb = nn.Embedding(num_embeddings = self.relation_count, embedding_dim = self.dim)

        torch.nn.init.xavier_uniform_(relations_emb.weight.data)

        return relations_emb

    def _init_attention_embedding(self):

        w = torch.empty(2*self.dim, self.dim)
        torch.nn.init.xavier_uniform_(w)
        return torch.nn.Parameter(w, requires_grad = True)

    
    def forward(self, input_tensors):

        s, nbrs_s, r, candidates, nbrs_candidates, labels = input_tensors


        source_emb = self.entities_emb(s)
        relation_emb = self.relations_emb(r)
        candidates_emb = self.entities_emb(candidates)

        #nbrs_source_emb = self.init_entities_emb(nbrs_s)
        #print(nbrs_s.shape)
        #print(self.relations_emb.weight.shape)
        #print(self.train_graph.relation_vocab)
        #print(self.train_graph.entity_vocab)


        nbrs_rel_emb = self.relations_emb(nbrs_s[:, :, 0])
        nbrs_ent_emb = self.entities_emb(nbrs_s[:, :, 1])
        nbrs_source_emb = (nbrs_rel_emb, nbrs_ent_emb)
        mask_nbrs_s = (~nbrs_s[:, :, 1].eq(self.train_graph.ent_pad)).type(torch.FloatTensor).to('cuda:0')

    

    # Perform attention to construct source feature vectors
        source_vec = self.attend(
            source_emb, nbrs_source_emb, relation_emb, mask_nbrs_s, name="source"
        )

        # Score candidates
        source_dot_query = source_vec * relation_emb
        scores = torch.squeeze(
                torch.matmul(
                    torch.unsqueeze(source_dot_query, 1), torch.transpose(candidates_emb, 1,2)), dim=1)
        #print(scores.shape)
        #print(labels.shape)
        if self.is_train:
            loss = self.loss(scores, labels)
            return scores, loss
        else:
            return scores
    
        
    def loss(self, logits, labels):
        batch_loss = nn.functional.cross_entropy(logits, labels)
        return batch_loss
    
    def distance(self, triples):

        assert triples.size()[1] == 3
        heads = triples[:,0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        return torch.sum(self.entities_emb(heads) * self. relations_emb(relations) * self.entities_emb(tails), -1).flatten()

    def attend(self, node, neighbors, query, nbr_mask, name=""):

        nbrs_rels, nbrs_ents = neighbors
        nbr_rel_concat = torch.cat((nbrs_rels, nbrs_ents), axis=-1)
        nbr_emb = torch.matmul(nbr_rel_concat, self.attention_emb_n)
        #print(nbr_emb.shape)

        node_query = torch.unsqueeze(node * query, 1)
        #node_query = torch.cat((node, query), axis=-1)
        #print(node_query.shape)
        #node_emb = torch.matmul(node_query, nbr_emb)
        #print(node_emb.shape)
        #node_emb = torch.unsqueeze(node_emb, 1)
        #print(node_emb.shape)
        #print(neighbors.shape)
        nbr_scores = torch.squeeze(torch.matmul(node_query, torch.transpose(nbr_emb,1,2)), dim=1)
        #nbr_scores = torch.matmul(node_emb, torch.transpose(neighbors,1,2))

        # mask out non-existing neighbors by adding a large negative number
        nbr_scores += (1 - nbr_mask) * (-1e7)
        attention_probs = torch.squeeze(self.softmax(nbr_scores))

        attention_emb = torch.sum(
            torch.unsqueeze(attention_probs, -1) * nbr_emb, 1
        )
        # Now concat attention_emb with node embedding and then project to emb_dim
        concat_emb = torch.cat((node, attention_emb), -1)
        output_emb = torch.matmul(concat_emb, self.attention_emb_s)
        
        if self.is_train:
            output = self.dropout(output_emb)
        else:
            output = output_emb
        return output


class DistMult(nn.Module):

    def __init__(self, train_graph, is_train, train_dropout = 0.5, dim = 100, norm = 1, margin = 1.0, epsilon = None):

        super(DistMult, self).__init__()

        self.entity_count = train_graph.ent_vocab_size
        self.relation_count = train_graph.rel_vocab_size
        self.dim = dim
        self.model = {}
        self.is_train = is_train
        self.dropout_p = train_dropout
        self.train_graph = train_graph
        self.norm = norm
        self.epsilon = epsilon

        self.entities_emb = self._init_entity_emb(margin, self.epsilon)
        self.model['entities_emb'] = self.entities_emb

        # self.init_entities_emb = self.entities_emb
        # self.model['init_entities_emb'] = self.init_entities_emb

        self.relations_emb = self._init_relation_emb(margin, self.epsilon)
        self.model['relations_emb'] = self.entities_emb

        # self.attention_emb_n = self._init_attention_embedding()
        # self.model['attention_emb_n'] = self.attention_emb_n

        # self.attention_emb_s = self._init_attention_embedding()
        # self.model['attention_emb_s'] = self.attention_emb_s

        #self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        #self.criterion = nn.MarginRankingLoss(margin = margin)


        #self.criterion = nn.MarginRankingLoss(margin = margin, reduction='none')

    def _init_entity_emb(self, margin, epsilon):
      
      entities_emb = nn.Embedding(num_embeddings = self.entity_count + 1, embedding_dim = self.dim, padding_idx=self.entity_count)

      if margin == None or epsilon == None:
        torch.nn.init.xavier_uniform_(entities_emb.weight.data)
        
      else:
        self.embedding_range = torch.nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False)
        torch.nn.init.uniform_(tensor = entities_emb.weight.data, a = -self.embedding_range.item(), b = self.embedding_range.item())
      return entities_emb

    def _init_relation_emb(self, margin, epsilon):
      
      relations_emb = nn.Embedding(num_embeddings = self.relation_count +1, embedding_dim = self.dim, padding_idx=self.relation_count)

      if margin == None or epsilon == None:
        torch.nn.init.xavier_uniform_(relations_emb.weight.data)
      else:
        self.embedding_range = torch.nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False)
        torch.nn.init.uniform_(tensor = relations_emb.weight.data, a = -self.embedding_range.item(), b = self.embedding_range.item())

      return relations_emb

    # def _init_attention_embedding(self):

    #     w = torch.empty(2*self.dim, self.dim)
    #     torch.nn.init.xavier_uniform_(w)
    #     return torch.nn.Parameter(w, requires_grad = True)

    
    def forward(self, input_tensors):

        s, nbrs_s, r, candidates, nbrs_candidates, labels = input_tensors


        source_emb = self.entities_emb(s)
        relation_emb = self.relations_emb(r)
        candidates_emb = self.entities_emb(candidates)

        #nbrs_source_emb = self.init_entities_emb(nbrs_s)
        #print(nbrs_s.shape)
        #print(self.relations_emb.weight.shape)
        #print(self.train_graph.relation_vocab)
        #print(self.train_graph.entity_vocab)


        # nbrs_rel_emb = self.relations_emb(nbrs_s[:, :, 0])
        # nbrs_ent_emb = self.entities_emb(nbrs_s[:, :, 1])
        # nbrs_source_emb = (nbrs_rel_emb, nbrs_ent_emb)
        # mask_nbrs_s = (~nbrs_s[:, :, 1].eq(self.train_graph.ent_pad)).type(torch.FloatTensor)

    

    # Perform attention to construct source feature vectors
        # source_vec = self.attend(
        #     source_emb, nbrs_source_emb, relation_emb, mask_nbrs_s, name="source"
        # )

        # Score candidates
        source_dot_query = source_emb * relation_emb
        scores = torch.squeeze(
                torch.matmul(
                    torch.unsqueeze(source_dot_query, 1), torch.transpose(candidates_emb, 1,2)), dim=1)
        #print(scores.shape)
        #print(labels.shape)
        if self.is_train:
            loss = self.loss(scores, labels)
            return scores, loss
        else:
            return scores
    
        
    def loss(self, logits, labels):
        batch_loss = nn.functional.cross_entropy(logits, labels)
        return batch_loss
    

class TransE(nn.Module):

    def __init__(self, train_graph, is_train, train_dropout = 0.5, dim = 100, norm = 1, margin = 1.0, epsilon = None):

        super(TransE, self).__init__()

        self.entity_count = train_graph.ent_vocab_size
        self.relation_count = train_graph.rel_vocab_size
        self.dim = dim
        self.model = {}
        self.is_train = is_train
        self.dropout_p = train_dropout
        self.train_graph = train_graph
        self.norm = norm
        self.epsilon = epsilon

        self.entities_emb = self._init_entity_emb(margin, self.epsilon)
        self.model['entities_emb'] = self.entities_emb

        # self.init_entities_emb = self.entities_emb
        # self.model['init_entities_emb'] = self.init_entities_emb

        self.relations_emb = self._init_relation_emb(margin, self.epsilon)
        self.model['relations_emb'] = self.entities_emb

        # self.attention_emb_n = self._init_attention_embedding()
        # self.model['attention_emb_n'] = self.attention_emb_n

        # self.attention_emb_s = self._init_attention_embedding()
        # self.model['attention_emb_s'] = self.attention_emb_s

        #self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        #self.criterion = nn.MarginRankingLoss(margin = margin)


        #self.criterion = nn.MarginRankingLoss(margin = margin, reduction='none')

    def _init_entity_emb(self, margin, epsilon):
      
      entities_emb = nn.Embedding(num_embeddings = self.entity_count, embedding_dim = self.dim)

      if margin == None or epsilon == None:
        torch.nn.init.xavier_uniform_(entities_emb.weight.data)
        
      else:
        self.embedding_range = torch.nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False)
        torch.nn.init.uniform_(tensor = entities_emb.weight.data, a = -self.embedding_range.item(), b = self.embedding_range.item())
      return entities_emb

    def _init_relation_emb(self, margin, epsilon):
      
      relations_emb = nn.Embedding(num_embeddings = self.relation_count, embedding_dim = self.dim)

      if margin == None or epsilon == None:
        torch.nn.init.xavier_uniform_(relations_emb.weight.data)
      else:
        self.embedding_range = torch.nn.Parameter(torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False)
        torch.nn.init.uniform_(tensor = relations_emb.weight.data, a = -self.embedding_range.item(), b = self.embedding_range.item())

      return relations_emb

    # def _init_attention_embedding(self):

    #     w = torch.empty(2*self.dim, self.dim)
    #     torch.nn.init.xavier_uniform_(w)
    #     return torch.nn.Parameter(w, requires_grad = True)

    
    def forward(self, input_tensors):

        s, nbrs_s, r, candidates, nbrs_candidates, labels = input_tensors

        #print(s.shape)
        #print(torch.unsqueeze(s,1).shape)
        #print(r.shape)
        #print(torch.unsqueeze(r,1).shape)
        #print(candidates.shape)
        source_emb = self.entities_emb(s)
        relation_emb = self.relations_emb(r)
        candidates_emb = self.entities_emb(candidates)

        #nbrs_source_emb = self.init_entities_emb(nbrs_s)
        #print(source_emb.shape)
        source_emb = torch.unsqueeze(source_emb,1)
        #print(relation_emb.shape)
        relation_emb = torch.unsqueeze(relation_emb,1)
        #print(candidates_emb.shape)
        #print(self.train_graph.relation_vocab)
        #print(self.train_graph.entity_vocab)


        # nbrs_rel_emb = self.relations_emb(nbrs_s[:, :, 0])
        # nbrs_ent_emb = self.entities_emb(nbrs_s[:, :, 1])
        # nbrs_source_emb = (nbrs_rel_emb, nbrs_ent_emb)
        # mask_nbrs_s = (~nbrs_s[:, :, 1].eq(self.train_graph.ent_pad)).type(torch.FloatTensor)

    

    # Perform attention to construct source feature vectors
        # source_vec = self.attend(
        #     source_emb, nbrs_source_emb, relation_emb, mask_nbrs_s, name="source"
        # )

        # Score candidates
        
        #source_dot_query = source_emb * relation_emb
        # scores = torch.squeeze(
        #         torch.matmul(
        #             torch.unsqueeze(source_dot_query, 1), torch.transpose(candidates_emb, 1,2)), dim=1)
        #print(scores.shape)
        #print(labels.shape)
        scores = (source_emb + relation_emb) - candidates_emb
        
        scores = torch.norm(scores, p=1, dim=-1)

        if self.is_train:
            loss = self.loss(scores, labels)
            return scores, loss
        else:
            return scores
    
        
    def loss(self, logits, labels):
        batch_loss = nn.functional.cross_entropy(logits, labels)
        return batch_loss
    
