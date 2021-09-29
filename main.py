from absl import app
from absl import flags
from tqdm import tqdm


import torch
from graph import Graph
from dataset import MyDataset
from models import Attention, TransE, DistMult
from metrics import hits_at_k, mrr_calc
from torch.utils import data
import tensorflow as tf

def evaluate(model_path, add_inverse_edge , device, train_kg_file = None, val_kg_file = None, test_kg_file = None):
    """Run evaluation on dev or test data."""
    
    train_graph = Graph(
        kg_file = train_kg_file
    )

    entity_vocab = train_graph.entity_vocab
    relation_vocab = train_graph.relation_vocab

    val_graph = None
    
    if val_kg_file!=None:
        val_graph = Graph(
            kg_file= val_kg_file,
            entity_vocab=entity_vocab,
            relation_vocab= relation_vocab,
            mode="val"
        )
    test_graph = None
    if test_kg_file!=None:
        test_graph = Graph(
            kg_file= test_kg_file,
            entity_vocab=entity_vocab,
            relation_vocab= relation_vocab,
            mode="test"
        )

    if not val_kg_file and not test_kg_file:
        raise ValueError("Evalution without a val or test file!")
    val_set = MyDataset(data_graph=val_graph, max_negatives=2000, max_neighbours=300, add_inverse=False, train_graph=train_graph, val_graph=val_graph, mode = 'val')
    val_loader = data.DataLoader(val_set, batch_size=128)

    
    model = Attention(train_graph = train_graph, dim = 256, is_train = False)

    model.to('cuda:0')
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)


    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
   #all_hits_summaries = []
    #all_mrr_summaries = []
    #print(val_graph.tuple_store)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr = 0
    batch_count = 0
    for s, nbrs_s, r, candidates, nbrs_candidates, labels in val_loader:

        s, nbrs_s, r, candidates, nbrs_candidates = (s.to(device), nbrs_s.to(device), r.to(device),
                                                             candidates.to(device), nbrs_candidates.to(device))
        candidate_scores = model((s, nbrs_s, r, candidates, nbrs_candidates, None))

    # Create eval metrics
    # if FLAGS.dev_kg_file:
        batch_rr = mrr_calc(candidate_scores, candidates, labels)
        mrr += torch.mean(batch_rr)
        #mrr_summary = tf.summary.scalar("MRR", mrr)
        #all_mrr_summaries.append(mrr_summary)
        
        #print("MRR: ", mrr)
        

        hits_at_1 += torch.mean(hits_at_k(candidate_scores, candidates, labels, k=1))
        hits_at_3 += torch.mean(hits_at_k(candidate_scores, candidates, labels, k=3))
        hits_at_10 += torch.mean(hits_at_k(candidate_scores, candidates, labels, k=10))

        #hits = torch.mean(batch_hits)
        #hits_summary = tf.summary.scalar("Hits_at_%d" % k, hits)
        #all_hits.append(hits)
        #all_hits_update.append(hits_update)
        #all_hits_summaries.append(hits_summary)
        #print("Hits_at_%d:" % 10, hits)
        # hits = tf.group(*all_hits)
        # hits_update = tf.group(*all_hits_update)
        batch_count +=1
    hits_at_1_score = (hits_at_1/batch_count)*100
    hits_at_3_score = (hits_at_3/batch_count)*100
    hits_at_10_score = (hits_at_10/batch_count)*100
    mrr_score = (mrr/batch_count)*100

    print( hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score)
    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score







def main(argv):



  

if __name__ == '__main__':
  app.run(main)