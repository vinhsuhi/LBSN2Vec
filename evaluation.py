import scipy.io
from scipy.io import loadmat
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10):
    """
    embs_user: np array shape NxD
    """
    embs_user = embs_user / np.linalg.norm(embs_user, axis=1, keepdims=True)
    scores = embs_user.dot(embs_user.T)
    # not evaluate friendship old
    mask_friendship_old = np.zeros_like(scores, dtype=np.bool)
    mask_friendship_old[friendship_old[:,0], friendship_old[:,1]] = True
    mask_friendship_old = mask_friendship_old + mask_friendship_old.T
    scores *= ~mask_friendship_old
    # lower diagonal matrix
    scores = np.tril(scores, k=-1)
    # rank scores
    inds = np.argwhere(scores > 0)
    rank_list = np.zeros((inds.shape[0], 3))
    rank_list[:, :2] = inds
    rank_list[:,2] = scores[inds[:,0], inds[:, 1]]
    rank_list = rank_list[np.argsort(-rank_list[:, 2])]
    # select topK 
    rank_list = rank_list[:k]
    selected_links = np.zeros_like(scores, dtype=np.bool)
    selected_links[rank_list[:,0].astype(np.int), rank_list[:,1].astype(np.int)] = True
    selected_links = selected_links + selected_links.T
    mask_friendship_new = np.zeros_like(scores, dtype=np.bool)
    mask_friendship_new[friendship_new[:,0], friendship_new[:,1]] = 1
    mask_friendship_new = mask_friendship_new + mask_friendship_new.T

    precision = precision_score(mask_friendship_new, selected_links, average='micro')
    recall = recall_score(mask_friendship_new, selected_links, average='micro')
    f1 = f1_score(mask_friendship_new, selected_links, average='micro')
    accuracy = accuracy_score(mask_friendship_new, selected_links)
    print(f"Precision@{k}: {precision:.3f}")
    print(f"Recall@{k}: {recall:.3f}")
    print(f"F1@{k}: {f1:.3f}")
    print(f"Accuracy@{k}: {f1:.3f}")


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

