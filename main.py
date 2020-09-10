import numpy as np
import networkx as nx
from tqdm import tqdm
import pdb
import os
from evaluation import *
import argparse
from embedding_model import EmbModel
import torch
import time
from utils import save_info, load_data, sample_neg, read_embs, initialize_emb, random_walk


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--mobility_ratio', type=float, default=0.2)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) # ten times for python would return better result
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--mode', type=str, default='friend', help="friend or POI")
    parser.add_argument('--input_type', type=str, default="mat", help="mat or persona") 
    parser.add_argument('--load', action='store_true') 
    parser.add_argument('--py', action='store_true') 
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--clean', action='store_true', help='use cleaned dataset')
    parser.add_argument('--batchsize', type=int, default=512)
    args = parser.parse_args()
    return args


def train_social(embedding_model, optimizer, win_size, alpha, this_sentences, \
                    j, loss1s, min_user, max_user, num_neg):
    edges = []
    for k in range(1, win_size + 1):
        if np.random.rand() > alpha:
            if j >= k:
                this_edges1 = this_sentences[:, [j, j - k]]
                edges.append(this_edges1)
            if j + k < len(this_sentences):
                this_edges2 = this_sentences[:, [j, j + k]]
                edges.append(this_edges2)
    if len(edges) > 0:
        edges = np.concatenate(edges, axis=0)
        edges = torch.LongTensor(edges)
        edges = edges.cuda()
        neg = np.random.randint(min_user, max_user, num_neg)
        neg = torch.LongTensor(neg).cuda()
        optimizer.zero_grad()
        loss1 = embedding_model.edge_loss(edges, neg)
        loss1s.append(loss1.item())
        loss1.backward()
        optimizer.step()


def train_poi(user_checkins_dict, this_sentences, embedding_model, loss2s, alpha, optimizer, j):
    words = this_sentences[:, j]
    this_user_checkins = []
    for w in words:
        try:
            this_checkins = user_checkins_dict[w]
            this_user_checkins.append(this_checkins)
        except:
            continue
    this_user_checkins = np.concatenate(this_user_checkins, axis=0)
    num_checkins_to_sample = int(alpha * len(this_user_checkins))
    if num_checkins_to_sample > 0:
        checkin_indices = np.random.randint(0, len(this_user_checkins), min(num_checkins_to_sample, 2*win_size*BATCH_SIZE))
        checkins = this_user_checkins[checkin_indices]
        checkins = torch.LongTensor(checkins).cuda()
        neg_ind = np.random.randint(0, len(selected_checkins), 10)
        neg = selected_checkins[neg_ind]
        neg = torch.LongTensor(neg).cuda()
        optimizer.zero_grad()
        loss2 = embedding_model.hyperedge_loss(checkins, neg)
        loss2s.append(loss2.item())
        loss2.backward()
        optimizer.step()


def train_persona(embedding_model, optimizer, maps, new_maps, this_sentences, j, min_user, max_user, num_neg):
    words = this_sentences[:, j]
    groups = [maps[ele] for ele in words]
    toconnect = np.array([np.random.choice(ele) for ele in groups])
    edges = np.array([words, toconnect])
    edges = torch.LongTensor(edges).cuda()
    neg = np.random.randint(min_user, max_user, num_neg)
    neg = torch.LongTensor(neg).cuda()
    optimizer.zero_grad()
    loss_persona = embedding_model.edge_loss(edges, neg)
    loss_persona.backward()
    optimizer.step()


def learn_emb(args, sentences, n_nodes, emb_dim, n_epochs, win_size, \
        selected_checkins, user_checkins_dict, alpha=0.2, num_neg=10, maps=None, new_maps=None):
    if maps is not None:
        mymaps = {k - 1: v - 1 for k, v in maps.items()}
        mynew_maps = {k - 1: [ele - 1 for ele in v] for k, v in new_maps.items()}
    min_user = np.min(selected_checkins[:,0])
    max_user = np.max(selected_checkins[:,0])
    sentences = np.array(sentences)
    embedding_model = EmbModel(n_nodes, emb_dim)
    embedding_model = embedding_model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=args.learning_rate)
    sentence_length = sentences.shape[1]
    BATCH_SIZE = args.batchsize
    N_ITERS = len(sentences) // BATCH_SIZE
    if N_ITERS % BATCH_SIZE > 0:
        N_ITERS += 1
    for _ in range(n_epochs):
        np.random.shuffle(sentences)
        for iter in tqdm(range(N_ITERS)):
            this_sentences = sentences[iter * BATCH_SIZE: (iter + 1) * BATCH_SIZE]
            loss1s = []
            loss2s = []
            for j in range(sentence_length):
                train_social(embedding_model, optimizer, win_size, \
                    alpha, this_sentences, j, loss1s, min_user, max_user, num_neg)
                train_poi(user_checkins_dict, this_sentences, embedding_model, loss2s, alpha, optimizer, j)
                if maps is not None: 
                    train_persona(embedding_model, optimizer, mymaps, mynew_maps, this_sentences, j, min_user, max_user, num_neg)
            print("loss1: {:.4f}".format(np.mean(loss1s)))
            print("loss2: {:.4f}".format(np.mean(loss2s)))
            print("-"*100)
    embeddings = embedding_model.node_embedding(torch.LongTensor(np.arange(n_nodes)).cuda())
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings


if __name__ == "__main__":
    args = parse_args()
    train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps = load_data(args)
    if not args.load:
        sentences = random_walk(friendship_old, n_users, args)
        if not args.py:
            import learn
            neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
            embs_ini = initialize_emb(args, n_nodes_total)
            save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)
            
            learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
            embs_file = "temp/processed/embs.txt"
            embs = read_embs(embs_file)
        else:
            train_checkins -= 1
            val_checkins -= 1
            train_user_checkins = {key - 1: value - 1 for key, value in train_user_checkins.items()}
            val_user_checkins = {key - 1: value - 1 for key, value in val_user_checkins.items()}
            friendship_new -= 1
            friendship_old -= 1
            for i in range(len(sentences)):
                sentences[i] = [x-1 for x in sentences[i]]
            embs = learn_emb(args, sentences, n_nodes_total, args.dim_emb, args.num_epochs, args.win_size, \
                train_checkins, train_user_checkins, alpha=args.mobility_ratio, num_neg = args.K_neg, maps=maps, new_maps=new_maps)
    else:
        embs_file = "temp/processed/embs.txt"
        embs = read_embs(embs_file)

    # evaluate
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]

    if args.mode == 'friend':
        if not args.py:
            friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10, new_maps=new_maps, maps=maps)
        else:
            friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=new_maps, maps=maps)
    else:
        val_checkins[:,0] -= 1
        val_checkins[:,1] -= (offset1+1)
        val_checkins[:,2] -= (offset2+1)
        location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
