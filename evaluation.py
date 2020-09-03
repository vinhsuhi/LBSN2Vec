import scipy.io
from scipy.io import loadmat
import numpy as np 
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def normalize_embedding(emb):
    normalize_factor = np.sqrt((emb ** 2).sum(axis=1))
    return emb / normalize_factor.reshape(-1, 1)


def friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=None, maps=None):
    normalized_embs_user = normalize_embedding(embs_user)
    simi_matrix = normalized_embs_user.dot(normalized_embs_user.T)
    for i in range(len(simi_matrix)):
        simi_matrix[i, i] = -2
    for i in range(len(friendship_old)):
        friendship_i = friendship_old[i]
        simi_matrix[friendship_i[0], friendship_i[1]] = -2
        simi_matrix[friendship_i[1], friendship_i[0]] = -2
    arg_sorted_simi = simi_matrix.argsort(axis=1)

    # sorted_simi = np.sort(simi_matrix, axis=1)[-10:]
    n_relevants = 0
    for i in tqdm(range(len(friendship_new))):
        first_node = friendship_new[i][0]
        second_node = friendship_new[i][1]
        if new_maps is not None:
            first_group = new_maps[first_node]
            second_group = new_maps[second_node]
            for ele in first_group:
                line_ele = arg_sorted_simi[ele]
                count = 0
                group = []
                flag = 0
                # duyet tu cuoi ve dau (10 groups)
                for kk in range(1, len(line_ele)):
                    target_index = line_ele[-kk]
                    group_target_index = maps[target_index]
                    if group_target_index not in group:
                        group.append(group_target_index)
                        count += 1
                        if count == k:
                            break 
                    if target_index in second_group:
                        n_relevants += 1
                        flag = 1
                        break 
                if flag == 1:
                    break
        
        else:
            line_ele = arg_sorted_simi[first_node]
            if second_node in line_ele[-k:]:
                n_relevants += 1
        
    precision = n_relevants / len(friendship_new)
    print(f"Precision@{k}: {precision:.3f}")


def friendship_linkprediction2(embs_user, friendship_old, friendship_new, k=10, new_maps=None):
    friendship_old_dict = {(x[0], x[1]): True for x in friendship_old}
    friendship_new_dict = {(x[0], x[1]): True for x in friendship_new if (x[0], x[1]) not in friendship_old_dict and (x[1], x[0]) not in friendship_old_dict}
    scores = embs_user.dot(embs_user.T)
    scores = np.tril(scores, k=-1)# lower diagonal matrix
    # scores[scores < 0.5] = 0
    # scores[not_trained_user_ids, :] = 0
    # scores[:, not_trained_user_ids] = 0
    scores[friendship_old[:,0], friendship_old[:,1]] = 0 # evaluate only new friendship
    scores[friendship_old[:,1], friendship_old[:,0]] = 0
    # rank scores
    inds = np.argwhere(scores > 0)
    rank_list = np.zeros((inds.shape[0], 3))
    rank_list[:, :2] = inds
    rank_list[:,2] = scores[inds[:,0], inds[:, 1]]
    rank_list = rank_list[np.argsort(-rank_list[:, 2])]
    import pdb 
    pdb.set_trace()
    n_relevants = 0
    for src, trg, score in rank_list[:k]:
        if (src, trg) in friendship_new_dict or (trg, src) in friendship_new_dict:
            n_relevants += 1
    precision = n_relevants/k
    recall = n_relevants/len(friendship_new_dict)
    # f1 = 2*precision*recall/(precision+recall)
    print(f"Precision@{k}: {precision:.3f}")
    print(f"Recall@{k}: {recall:.3f}")
    # print(f"F1@{k}: {np.mean(f1s):.3f}")

def friendship_linkprediction_with_sample(embs_user, friendship_old, friendship_new, k=10):
    friendship_old_dict = {(x[0], x[1]): True for x in friendship_old}
    friendship_new_dict = {(x[0], x[1]): True for x in friendship_new if (x[0], x[1]) not in friendship_old_dict 
                            and (x[1], x[0]) not in friendship_old_dict
                                #and x[0] in trained_user_ids and x[1] in trained_user_ids
                        }
    scores = embs_user.dot(embs_user.T)
    scores = np.tril(scores, k=-1)# lower diagonal matrix
    # scores[scores < 0.5] = 0
    # scores[not_trained_user_ids, :] = 0
    # scores[:, not_trained_user_ids] = 0
    scores[friendship_old[:,0], friendship_old[:,1]] = 0 # evaluate only new friendship
    scores[friendship_old[:,1], friendship_old[:,0]] = 0
    # rank scores
    inds = np.argwhere(scores > 0)
    rank_list = np.zeros((inds.shape[0], 3))
    rank_list[:, :2] = inds
    rank_list[:,2] = scores[inds[:,0], inds[:, 1]]
    rank_list = rank_list[np.argsort(-rank_list[:, 2])]

    precisions = []
    recalls = []
    f1s = []

    for i in range(10):
        # select 1%
        n_select = int(len(rank_list)*0.01)
        selected_rank_list = rank_list[np.random.choice(len(rank_list), n_select)]
        # select topk
        selected_rank_list = selected_rank_list[:k]

        n_relevants = 0
        for src, trg, score in selected_rank_list:
            if (src, trg) in friendship_new_dict or (trg, src) in friendship_new_dict:
                n_relevants += 1
        precision = n_relevants/k
        recall = n_relevants/len(friendship_new_dict)
    #     f1 = 2*precision*recall/(precision+recall)
        precisions.append(precision)
        recalls.append(recall)
    #     f1s.append(f1)
    print(f"Precision@{k}: {np.mean(precisions):.3f}")
    print(f"Recall@{k}: {np.mean(recall):.3f}")
    # print(f"F1@{k}: {np.mean(f1s):.3f}")


def location_prediction(test_checkin, embs, poi_embs, k=10):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    correct = 0
    for user, timeslot, poi in tqdm(test_checkin):
        user_emb = embs[user]
        time_emb = embs[timeslot]
        scores = np.sum(user_emb*poi_embs, axis=1) + np.sum(time_emb*poi_embs, axis=1)
        pred_pois = np.argsort(-scores)[:k]
        if poi in pred_pois:
            correct += 1
    try:
        acc = correct/ len(test_checkin)
    except:
        import pdb
        pdb.set_trace()
    print(f"Accuracy@{k}: {acc:.3f}")


def loadtxt(path, separator):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split(separator)
            data.append([float(ele) for ele in data_line])
    return np.array(data)


if __name__ == "__main__":
    embs_user = np.random.uniform(size=(4, 5))
    fo = np.array([[0,1], [2, 3]])
    fn = np.array([[0,2], [0,3], [1, 2]])
    friendship_linkprediction(embs_user, fo, fn)

    # embs_cate = loadtxt('embs_cate.txt', ',').T 
    # embs_user = loadtxt('embs_user.txt', ',').T 
    # embs_time = loadtxt('embs_time.txt', ',').T 
    # embs_venue = loadtxt('embs_venue.txt', ',').T 

