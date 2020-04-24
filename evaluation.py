import scipy.io
from scipy.io import loadmat
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score

def friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10):
    """
    embs_user: np array shape NxD
    """
    embs_user = np.linalg.norm(embs_user)
    scores = embs_user.dot(embs_user.T)
    # not evaluate friendship old
    mask_friendship_old = np.ones((friendship_old.max(), friendship_old.max()))
    mask_friendship_old[friendship_old[:,0], friendship_old[:,1]] = 0
    scores *= mask_friendship_old
    # rank scores
    rank_list = np.dstack(np.unravel_index(np.argsort(scores.ravel()), scores.shape))
    # select topK 
    rank_list = rank_list[:k]
    selected_links = np.zeros_like(scores, dtype=np.int)
    selected_links[rank_list[:,0], rank_list[:,1]] = 1
    mask_friendship_new = np.zeros_like(scores, dtype=np.int)
    mask_friendship_new[friendship_new[:,0], friendship_new[:,1]] = 1

    precision = precision_score(mask_friendship_new, selected_links, average='micro')
    recall = recall_score(mask_friendship_new, selected_links, average='micro')
    f1 = f1_score(mask_friendship_new, selected_links, average='micro')
    print(f"Precision@{k}: {precision:.3f}")
    print(f"Recall@{k}: {recall:.3f}")
    print(f"F1-micro@{k}: {f1:.3f}")


def loadtxt(path, separator):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split(separator)
            data.append([float(ele) for ele in data_line])
    return np.array(data)

if __name__ == "__main__":
    embs_cate = loadtxt('embs_cate.txt', ',').T 
    embs_user = loadtxt('embs_user.txt', ',').T 
    embs_time = loadtxt('embs_time.txt', ',').T 
    embs_venue = loadtxt('embs_venue.txt', ',').T 

