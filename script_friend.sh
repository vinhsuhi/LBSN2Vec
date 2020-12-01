
######################## friend prediction #####################
    # copy POI
        # normal randomwalk

mode="persona_ori"

for data in Jakarta
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  
done



mode="persona_ori"

for lr in 0.0005 0.001
do 
for Neg in 5 10
do 
    for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
    do 
        python -u CMan.py --input_type ${mode} --dataset_name ${data} --K_neg ${Neg} > output/model2_${data}_friend_lr${lr}_N${Neg}
    done
done 
done 


    # tach POI (WAIT...)
        # normal random walk

mode="persona_POI"

for data in NYC
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  
done


mode="persona_POI"

for lr in 0.0005 0.001
do 
for Neg in 5 10
do 
    for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
    do 
        python -u CMan.py --input_type ${mode} --dataset_name ${data}  > output/model3_${data}_friend_lr${lr}_N${Neg}
    done
done 
done 


        # new random walk 

mode="persona_POI"

for data in NYC
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --alpha 0.1
done

mode="persona_POI"

for data in NYC
do 
        python -u CMan.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --q_n2v 0.8 --p_n2v 0.8 --test
done



mode="persona_POI"

for p in 0.8  
do 
for q in 0.6 
do 
for lr in 0.0005 0.001
do
for Neg in 5 10
do 
for dim in 256 300
do 
for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do
        python -u CMan.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --q_n2v ${q} --p_n2v ${p} --K_neg ${Neg} --dim_emb ${dim} > output/model4_${data}_friend_p${p}_q${q}_lr${lr}_N${Neg}_dim${dim}
done
done 
done
done 
done
done



    # Original_version


for data in NYC
do 
    python -u baselines.py --dataset_name ${data}  
done


for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do 
    python -u baselines.py --dataset_name ${data}  > output/original_${data}_friend
done


    # other baselines

# create input for those graphs


for dataset in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in dhne deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model} 
    done
done



# run embeddings 

for data in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do     
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}.embeddings
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}_M.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}_M.embeddings
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}_SM.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}_SM.embeddings
done

for data in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do 
python run_node2vec --dataset_name ${data}
python run_node2vec --dataset_name ${data}_M
python run_node2vec --dataset_name ${data}_SM
done

for data in Istanbul KualaLampur SaoPaulo NYC
do 
python run_node2vec --dataset_name ${data}
python run_node2vec --dataset_name ${data}_M
python run_node2vec --dataset_name ${data}_SM
done



for data in Istanbul KualaLampur SaoPaulo NYC
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_M.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_M.embeddings 
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_SM.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_SM.embeddings 
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_M.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_M.embeddings --epochs 2
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_SM.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_SM.embeddings --epochs 2
done

# Just M
for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do
python src/hypergraph_embedding.py --data_path ../LBSN2Vec/dhne_graph/${data} --save_path ../LBSN2Vec/dhne_emb/${data} -s 16 16 16 -b 256
done

# eval_embedding

for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in line 
    do 
        python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_friend
        python -u eval_models.py --emb_path ${model}_emb/${data}_M.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_M_friend
        python -u eval_models.py --emb_path ${model}_emb/${data}_SM.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_SM_friend
    done

    # model=dhne
    # python -u eval_models.py --emb_path ${model}_emb/${data}/model_16/embeddings.npy --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_friend

done 


python -m openne --method node2vec --label-file data/blogCatalog/bc_labels.txt --input data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output vec_all.txt --q 0.25 --p 0.25

