import scipy.io
from scipy.io import loadmat
import numpy as np 
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def normalize_embedding(emb):
    normalize_factor = np.sqrt((emb ** 2).sum(axis=1))
    return emb / normalize_factor.reshape(-1, 1)


def friendship_pred_ori(embs, friendship_old, friendship_new, k=10):
    """
    Simplest Linkprediction Evaluation
    embs: user embeddings
    friendship_old: old friendship: node_id >= 1
    friendship_new: new friendship: node_id >= 1
    """

    friendship_old -= 1
    friendship_new -= 1
    ################# compute simi matrix #################
    num_users = embs.shape[0]
    normalize_embs = normalize_embedding(embs)
    simi_matrix = normalize_embs.dot(normalize_embs.T)
    #######################################################
    

    ################# preprocess simi matrix #########################
    for i in range(num_users):
        simi_matrix[i, i] = -2
    
    for i in range(friendship_old.shape[0]):
        simi_matrix[friendship_old[i, 0], friendship_old[i, 1]] = -2
    ################################################################
    
    # argsort
    arg_sorted_simi = simi_matrix.argsort(axis=1)
    
    ################# create friend_dict: node-> set of fiends #############
    friend_dict = dict()
    for i in range(friendship_new.shape[0]):
        source, target = friendship_new[i][0], friendship_new[i][1]
        if source not in friend_dict:
            friend_dict[source] = set([target])
        else:
            friend_dict[source].add(target)
    ########################################################################

    ###################### evaluate #########################
    for kk in [10, 20, 50, 100, 200]:
        precision_k, recall_k = compute_precision_recall(friend_dict, arg_sorted_simi, kk)
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        # print(f"Precision@{kk}: {precision_k:.3f}")
        # print(f"Recall@{kk}: {recall_k:.3f}")
        print("{:.4f}|{:.4f}".format(precision_k, recall_k))
        # print(f"F1@{kk}: {f1_k:.3f}")
    #########################################################


def compute_precision_recall(friend_dict, arg_sorted_simi, k):
    precision = []
    recall = []
    for key, value in friend_dict.items():
        n_relevants = 0
        arg_simi_key = arg_sorted_simi[key][-k:]
        for target_node in value:
            if target_node in arg_simi_key:
                n_relevants += 1
        precision.append(n_relevants/k)
        recall.append(n_relevants/len(value))
    precision = np.mean(precision)
    recall = np.mean(recall)
    return precision, recall


def friendship_pred_persona(embs_user, friendship_old_ori, friendship_new, k=10, maps_OritP=None, maps_PtOri=None):
    normalized_embs_user = normalize_embedding(embs_user)
    simi_matrix = normalized_embs_user.dot(normalized_embs_user.T)
    for i in range(len(simi_matrix)):
        simi_matrix[i, i] = -2

    for i in range(len(friendship_old_ori)):
        friendship_i = friendship_old_ori[i]
        source, target = friendship_i[0], friendship_i[1]
        group_source = maps_OritP[source]
        group_target = maps_OritP[target]
        for persona_s in group_source:
            for persona_t in group_target:
                simi_matrix[persona_s - 1, persona_t - 1] = -2
                simi_matrix[persona_t - 1, persona_s - 1] = -2
    arg_sorted_simi = simi_matrix.argsort(axis=1)

    friend_dict = dict()
    for i in range(len(friendship_new)):
        source, target = friendship_new[i][0], friendship_new[i][1]
        if source not in friend_dict:
            friend_dict[source] = set([target])
        else:
            friend_dict[source].add(target)

    def is_match(ordered_candidates, target_gr, kk):
        group = []
        count = 0
        for i in range(1, len(ordered_candidates)):
            target_index = ordered_candidates[-i] + 1
            group_target_index = maps_PtOri[target_index]
            if group_target_index not in group:
                group.append(group_target_index)
                count += 1
                if count == kk + 1:
                    break
            if target_index in target_gr:
                return 1
        return 0

    def cal_precision_recall_k(kk):
        precision = []
        recall = []
        for user, friends_list in friend_dict.items():
            n_relevants = 0
            source_group = maps_OritP[user]
            target_groups = [maps_OritP[fr] for fr in friends_list]
            for persona_s in source_group:
                ordered_candidates = arg_sorted_simi[persona_s - 1]
                for j in range(len(target_groups)):
                    if is_match(ordered_candidates, target_groups[j], kk):
                        n_relevants += 1
                        target_groups[j] = []

            precision.append(n_relevants/kk)
            recall.append(n_relevants/len(friends_list))
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = 2 * precision * recall / (precision + recall)

        # print(f"Precision@{kk}: {precision:.3f}")
        # print(f"Recall@{kk}: {recall:.3f}")
        # print(f"F1@{kk}: {f1:.3f}")
        print("{:.4f}|{:.4f}".format(precision, recall))
    
    for kk in [10, 20, 50, 100, 200]:
        cal_precision_recall_k(kk)
        


def location_prediction(test_checkin, embs, poi_embs, k=10):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    embs = normalize_embedding(embs) # N x d
    poi_embs = normalize_embedding(poi_embs) # Np x d
    user_time = test_checkin[:, :2] # user and time 
    user_time_emb = embs[user_time] # n x 2 x d
    user_time_with_poi = np.dot(user_time_emb, poi_embs.T) # nx2x(np)
    user_time_with_poi = np.sum(user_time_with_poi, axis=1) # nxnp
    # argptt = np.argpartition(user_time_with_poi, -k, axis=1)[:, -k:] # nx10
    argptt = np.argsort(user_time_with_poi, axis=1)
    
    ranks = []
    hit10s = 0
    hit20s = 0
    hit30s = 0
    hit40s = 0
    hit50s = 0

    for i in range(argptt.shape[0]):
        rank = argptt.shape[1]
        for j in range(argptt.shape[1]):
            if test_checkin[i, 2] == argptt[i,j]:
                rank = j 
                if rank < 10:
                    hit10s += 1
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1
                elif rank < 20:
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1
                elif rank < 30:
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1
                elif rank < 40:
                    hit40s += 1
                    hit50s += 1
                elif rank < 50:
                    hit50s += 1
                break 
        ranks.append(rank + 1)

    try:
        mean_rank = np.mean(ranks)
        mrr = np.mean([1/ele for ele in ranks])
        hit10s /= len(test_checkin)
        hit20s /= len(test_checkin)
        hit30s /= len(test_checkin)
        hit40s /= len(test_checkin)
        hit50s /= len(test_checkin)
        print("Hit10: {:.4f}".format(hit10s))
        print("Hit20: {:.4f}".format(hit20s))
        print("Hit30: {:.4f}".format(hit30s))
        print("Hit40: {:.4f}".format(hit40s))
        print("Hit50: {:.4f}".format(hit50s))
        print("MR: {:.4f}".format(mean_rank))
        print("MRR: {:.4f}".format(mrr))
        # return acc
        return 1
    except Exception as err:
        print(err)
    

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



def rank_matrix(matrix, k=10):
    arg_max_index = []
    while len(arg_max_index) < 10:
        arg_max = np.argmax(matrix)
        column = arg_max % matrix.shape[1]
        arg_max_index.append(column)
        matrix[:, column] -= 2
    return arg_max_index


def location_prediction_Persona(test_checkin, embs, poi_embs, k=10, user_persona_dict=None, persona_user_dict=None):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    embs = normalize_embedding(embs) # N x d
    poi_embs = normalize_embedding(poi_embs) # Np x d
    users = test_checkin[:, 0]
    times = test_checkin[:, 1]
    
    hit = 0
    for i, user in enumerate(users):
        this_user_persona = user_persona_dict[user + 1]
        this_user_persona = [ele - 1 for ele in this_user_persona]
        this_user_time = times[i]
        time_emb = embs[this_user_time].reshape(1, -1)
        time_ranking = time_emb.dot(poi_embs.T).reshape(1, -1)
        this_user_persona_emb = embs[this_user_persona]
        this_user_persona_ranking = this_user_persona_emb.dot(poi_embs.T).reshape(len(this_user_persona), -1)
        final_ranking = time_ranking + this_user_persona_ranking
        top_k = rank_matrix(final_ranking, k)
        target = test_checkin[i, 2]
        if target in top_k:
            hit += 1

    acc = hit / len(test_checkin)
    print(f"Accuracy@{k}: {acc:.3f}")
    return acc
    


def location_prediction_Persona2(test_checkin, embs, poi_embs, k=10, user_persona_dict=None, persona_user_dict=None):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    embs = normalize_embedding(embs) # N x d
    poi_embs = normalize_embedding(poi_embs) # Np x d
    users = test_checkin[:, 0]
    times = test_checkin[:, 1]
    
    ranks = []
    hit10s = 0
    hit20s = 0
    hit30s = 0
    hit40s = 0
    hit50s = 0

    for i, user in enumerate(users):
        this_user_persona = user_persona_dict[user + 1]
        this_user_persona = [ele - 1 for ele in this_user_persona]
        this_user_time = times[i]
        time_emb = embs[this_user_time].reshape(1, -1)
        time_ranking = time_emb.dot(poi_embs.T).reshape(1, -1)
        this_user_persona_emb = embs[this_user_persona]
        this_user_persona_ranking = this_user_persona_emb.dot(poi_embs.T).reshape(len(this_user_persona), -1)
        final_ranking = time_ranking + this_user_persona_ranking

        # argptt = np.argpartition(final_ranking, -k, axis=1)[:, -k:] # nx10
        argptt = np.argsort(final_ranking, axis=1)
        target = test_checkin[i, 2]
        rank = argptt.shape[1]
        flag = 0
        for j in range(argptt.shape[1]):
            for k in range(argptt.shape[0]):
                if argptt[k, j] == target:
                    rank = j 
                    if rank < 10:
                        hit10s += 1
                        hit20s += 1
                        hit30s += 1
                        hit40s += 1
                        hit50s += 1
                    elif rank < 20:
                        hit20s += 1
                        hit30s += 1
                        hit40s += 1
                        hit50s += 1
                    elif rank < 30:
                        hit30s += 1
                        hit40s += 1
                        hit50s += 1
                    elif rank < 40:
                        hit40s += 1
                        hit50s += 1
                    elif rank < 50:
                        hit50s += 1
                    flag = 1
                    break 
            if flag:
                break 
        ranks.append(rank + 1)
    try:
        # acc = hit / len(test_checkin)
        hit10s /= len(test_checkin)
        hit20s /= len(test_checkin)
        hit30s /= len(test_checkin)
        hit40s /= len(test_checkin)
        hit50s /= len(test_checkin)
        mean_rank = np.mean(ranks)
        mrr = np.mean([1/ele for ele in ranks])
        print("Hit10: {:.4f}".format(hit10s))
        print("Hit20: {:.4f}".format(hit20s))
        print("Hit30: {:.4f}".format(hit30s))
        print("Hit40: {:.4f}".format(hit40s))
        print("Hit50: {:.4f}".format(hit50s))
        print("MR: {:.4f}".format(mean_rank))
        print("MRR: {:.4f}".format(mrr))
        # return acc
    except:
        print("lol")
    


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
    # friendship_linkprediction(embs_user, fo, fn)

    # embs_cate = loadtxt('embs_cate.txt', ',').T 
    # embs_user = loadtxt('embs_user.txt', ',').T 
    # embs_time = loadtxt('embs_time.txt', ',').T 
    # embs_venue = loadtxt('embs_venue.txt', ',').T 

