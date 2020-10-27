
import argparse
from scipy.io import loadmat
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()
    return args


def read_input(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    friendship_old -= 1
    nodes = np.unique(friendship_old)
    print("Min: {}, Max: {}, Len: {}".format(np.min(nodes), np.max(nodes), np.len(nodes)))
    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old


def save_deepwalk(edges, model_name):
    out_dir = "edgelist_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    np.savetxt("{}/{}.edgeslist".format(out_dir, model_name), edges, delimiter='\t')
    print("Done!")


def save_line(edges, model_name):
    out_dir = "line_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open("{}/{}.edgeslist".format(out_dir, model_name), 'w', encoding='utf-8') as file:
        for i in range(edges.shape[0]):
            file.write("{}\t{}\t1\n".format(edges[i, 0], edges[i, 1]))
            file.write("{}\t{}\t1\n".format(edges[i, 1], edges[i, 0]))
    print("Done!")


def save_hebe(edges, model_name):
    out_dir = "hebe_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pass


def save_dhne(edges, model_name):
    out_dir = "dhne_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pass

# if __name__ == "__main__":
args = parse_args()
print(args)

model = args.model 
friendship = read_input(args.dataset_name)

if model.lower() == "deepwalk":
    save_deepwalk(friendship, args.dataset_name)
elif model.lower() == "node2vec":
    save_deepwalk(friendship, args.dataset_name)
elif model.lower() == "line":
    save_line(friendship, args.dataset_name)
elif model.lower() == "hebe":
    save_hebe(friendship, args.dataset_name)
elif model.lower() == "dhne":
    save_dhne(friendship, args.dataset_name)
else:
    print("Have not implement yet...")

"""

for dataset in NYC TKY hongzhi
do
    for model in deepwalk line
    do 
        python create_input.py --dataset_name ${dataset} --model ${model}
    done
done

"""