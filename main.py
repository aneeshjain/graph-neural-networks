from absl import app
from absl import flags
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import torch
from graph import Graph
from dataset import MyDataset
from models import Attention, TransE, DistMult
from metrics import hits_at_k, mrr_calc
from torch.utils import data
import tensorflow as tf
import sys
import os
import pandas as pd
import torch.nn as nn


FLAGS = flags.FLAGS


flags.DEFINE_float('lr', 2e-5, 'Learning Rate')
flags.DEFINE_integer('epochs', 5, 'Number of training epochs')
flags.DEFINE_string('output_path', './model_logs', 'Output path for model and logs')
flags.DEFINE_string('load_model_path', '', 'Path to load saved model for testing')



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


    checkpoint = torch.load(FLAGS.load_model_path)
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

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0')



    validation_freq = 1



    train_file_path = "./WN18RR/text/train.txt"
    val_file_path = "./WN18RR/text/valid.txt"
    test_file_path = "./WN18RR/text/test.txt"


    graph_type = Graph
    k_graph = graph_type(kg_file = train_file_path)

    data_set = MyDataset(data_graph=k_graph, max_negatives=2000, max_neighbours=300, add_inverse=False)
    train_loader = data.DataLoader(data_set, batch_size=1024)

    

    writer = SummaryWriter(FLAGS.output_path)


    model = Attention(train_graph = k_graph, dim = 256, is_train = True, train_dropout = 0.3)
    model = model.to(device)



    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    
    
    start_epoch_id = 1

    epochs = FLAGS.epochs



    for epoch in range(start_epoch_id, epochs+1):

        loss_epoch = 0

        model.train()
    
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            
            
            for s, nbrs_s, r, candidates, nbrs_candidates, labels in train_loader:

                s, nbrs_s, r, candidates, nbrs_candidates, labels = (s.to(device), nbrs_s.to(device), r.to(device), 
                                                                candidates.to(device), nbrs_candidates.to(device), labels.to(device))


                labels = torch.argmax(labels, 1)

                optimizer.zero_grad()
        
                score, loss = model((s, nbrs_s, r, candidates, nbrs_candidates, labels))

                loss.backward()
            
                optimizer.step()

                loss_epoch +=loss.item()
                
                
                tepoch.update(1)
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

            
            writer.add_scalar('Training loss',loss_epoch / 100, 
                epoch)
            
            
            
            print(f'Epoch {epoch}: {loss_epoch/100}')
            
            if epoch % validation_freq == 0:
            
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_epoch,
                }, FLAGS.output_path)

                model.eval()
                hits_at_1, hits_at_3, hits_at_10, mrr = evaluate(FLAGS.output_path, add_inverse_edge = False, device = device,
                                                            train_kg_file = train_file_path, val_kg_file = val_file_path, 
                                                            test_kg_file = None)
                writer.add_scalar('Metrics/Hits_1/' + 'val', hits_at_1, global_step=epoch)
                writer.add_scalar('Metrics/Hits_3/' + 'val', hits_at_3, global_step=epoch)
                writer.add_scalar('Metrics/Hits_10/' + 'val', hits_at_10, global_step=epoch)
                writer.add_scalar('Metrics/MRR/' + 'val', mrr, global_step=epoch)

                print(f'Hits@1: {hits_at_1}, Hits@3: {hits_at_3}, Hits@10: {hits_at_10}, MRR: {mrr}')

    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_epoch,
                }, FLAGS.output_path)
            
        

if __name__ == '__main__':
  app.run(main)