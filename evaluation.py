import scipy.io
from scipy.io import loadmat
import numpy as np 


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

