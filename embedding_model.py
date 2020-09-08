import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from loss import EmbeddingLossFunctions
from tqdm import tqdm

class EmbModel(nn.Module):
    def __init__(self, n_nodes, embedding_dim):
        super(EmbModel, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.n_nodes = n_nodes
        self.link_pred_layer = EmbeddingLossFunctions()

    def forward(self, nodes):
        node_output = self.node_embedding(nodes)
        # node_output = F.normalize(node_output, dim=1)
        return node_output

    def edge_loss(self, edges, neg):
        source_embedding = F.normalize(self.forward(edges[:, 0]), dim=1)
        target_embedding = F.normalize(self.forward(edges[:, 1]), dim=1)
        neg_embedding = F.normalize(self.forward(neg), dim=1)
        loss, loss0, loss1 = self.link_pred_layer.loss(source_embedding, target_embedding, neg_embedding)
        loss = loss/len(edges)
        return loss

    def hyperedge_loss(self, Nodes, negs):
        user_embs = self.forward(Nodes[0])
        time_embs = self.forward(Nodes[1])
        location_embs = self.forward(Nodes[2])
        cate_embs = self.forward(Nodes[3])

        user_means = torch.mean(user_embs, dim=0)
        time_means = torch.mean(time_embs, dim=0)
        loc_means = torch.mean(location_embs, dim=0)
        cate_means = torch.mean(cate_embs, dim=0)

        user_means = user_means.repeat(len(user_embs), 1)
        time_means = time_means.repeat(len(time_embs), 1)
        loc_means = loc_means.repeat(len(location_embs), 1)
        cate_means = cate_means.repeat(len(cate_embs), 1)

        neg_users = self.forward(negs[0])
        neg_times = self.forward(negs[1])
        neg_locs = self.forward(negs[2])
        neg_cates = self.forward(negs[3])

        loss0, _, _ = self.link_pred_layer.loss(user_means, user_embs, neg_users)
        loss1, _, _ = self.link_pred_layer.loss(time_means, time_embs, neg_times)
        loss2, _, _ = self.link_pred_layer.loss(loc_means, location_embs, neg_locs)
        loss3, _, _ = self.link_pred_layer.loss(cate_means, cate_embs, neg_cates)

        loss = loss0 + loss1 + loss2 + loss3
        loss = loss / len(user_embs)
        return loss



def learn_emb(sentences, n_nodes, emb_dim, n_epochs, win_size, \
        user_nodes, time_nodes, location_nodes, cate_nodes, user_checkins_dict, alpha=0.2, num_neg=10):
    min_user = np.min(user_nodes)
    max_user = np.max(user_nodes)
    min_time = np.min(time_nodes)
    max_time = np.max(time_nodes)
    min_location = np.min(location_nodes)
    max_location = np.max(location_nodes)
    min_cate = np.min(cate_nodes)
    max_cate = np.max(cate_nodes)

    embedding_model = EmbModel(n_nodes, emb_dim)
    embedding_model = embedding_model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=args.lr)

    for epoch in tqdm(range(n_epochs)):
        np.random.shuffle(sentences)
        for i in range(len(sentences)):
            this_sentence = sentences[i]
            for j in range(len(this_sentence)):
                word = this_sentence[j]
                edges = []
                for k in range(1, win_size + 1):
                    if np.random.rand() > alpha:
                        if j >= k:
                            target_e = this_sentence[j - k]
                            edges.append([word, target_e])
                        if j + k < len(this_sentence):
                            target_e = this_sentence[j + k]
                            edges.append([word, target_e])
                if len(edges) > 0:
                    edges = torch.LongTensor(np.array(edges))
                    edges = edges.cuda()
                    neg = np.random.randint(min_user, max_user, num_neg)
                    neg = torch.LongTensor(neg).cuda()
                    optimizer.zero_grad()
                    loss1 = embedding_model.edge_loss(edges, embedding_model, neg)
                    loss1.backward()
                    optimizer.step()
                    print("Loss1: {:.4f}".format(loss1))

                this_user_checkins = user_checkins_dict[word]
                if len(this_user_checkins) > 0:
                    sampled_users = []
                    sampled_times = []
                    sampled_locs = []
                    sampled_cates = []
                    for k in range(min(2 * win_size, len(this_user_checkins))):
                        if np.random.rand() < alpha:
                            checkin_index = np.random.randint(0, len(this_user_checkins), 1)[0]
                            checkin = this_user_checkins[checkin_index]
                            sampled_users.append(checkin[0])
                            sampled_times.append(checkin[1])
                            sampled_locs.append(checkin[2])
                            sampled_cates.append(checkin[3])
                    if len(sampled_users) > 0:
                        sampled_users = torch.LongTensor(np.array(sampled_users)).cuda()
                        sampled_times = torch.LongTensor(np.array(sampled_times)).cuda()
                        sampled_locs = torch.LongTensor(np.array(sampled_locs)).cuda()
                        sampled_cates = torch.LongTensor(np.array(sampled_cates)).cuda()
                        Nodes = [sampled_users, sampled_times, sampled_locs, sampled_cates]
                        neg_users = np.random.randint(min_user, max_user, num_neg)
                        neg_users = torch.LongTensor(neg_users).cuda()

                        neg_times = np.random.randint(min_time, max_time, num_neg)
                        neg_times = torch.LongTensor(neg_times).cuda()

                        neg_locs = np.random.randint(min_location, max_location, num_neg)
                        neg_locs = torch.LongTensor(neg_locs).cuda()

                        neg_cates = np.random.randint(min_cate, max_cate, num_neg)
                        neg_cates = torch.LongTensor(neg_cates).cuda()

                        negs = [neg_users, neg_times, neg_locs, neg_cates]

                        loss2 = embedding_model.hyperedge_loss(Nodes, negs)

                        


