
################################## PERSONA ######################################

# persona: For one, we need to run splitter to create persona graph
for DATA in Istanbul Jakarta KualaLampur Saopaulo TKY NYC hongzhi
do
    python src/main.py --edge-path input/${DATA}_friends.csv --lbsn ${DATA}
done 

# persona: then we just need to run LBSN with persona graphs
for DATA in Istanbul Jakarta KualaLampur Saopaulo TKY NYC hongzhi
do
    python -u main.py --dataset_name ${DATA} --mode POI --input_type persona > results/POI_persona_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type persona > results/friend_persona_${DATA}
done
hongzhi NYC TKY


for DATA in TKY
do
    python src/main.py --edge-path input/${DATA}_friends.csv --lbsn ${DATA}
done

# persona: then we just need to run LBSN with persona graphs
for DATA in  TKY
do
    python -u main.py --dataset_name ${DATA} --mode POI --input_type persona
    python -u main.py --dataset_name ${DATA} --mode friend --input_type persona
done

########################### PESONA POI ################################################
for DATA in hongzhi NYC TKY Istanbul Jakarta KualaLampur Saopaulo
do
python src/main.py --edge-path input/${DATA}_friendPOI.csv --lbsn ${DATA} --listPOI input/location_${DATA}
done


python src/main.py --edge-path input/hongzhi_friends.csv --lbsn hongzhi --listPOI input/location_hongzhi
python src/main.py --edge-path input/NYC_friends.csv --lbsn NYC --listPOI input/location_NYC
python src/main.py --edge-path input/TKY_friends.csv --lbsn TKY --listPOI input/location_TKY
python src/main.py --edge-path input/Istanbul_friends.csv --lbsn Istanbul --listPOI input/location_Istanbul
python src/main.py --edge-path input/Jakarta_friends.csv --lbsn Jakarta --listPOI input/location_Jakarta
python src/main.py --edge-path input/KualaLampur_friends.csv --lbsn KualaLampur --listPOI input/location_KualaLampur
python src/main.py --edge-path input/Saopaulo_friends.csv --lbsn Saopaulo --listPOI input/location_Saopaulo

########################### PESONA FRIEND , SPLIT POI ################################################
python src/main.py --edge-path input/hongzhi_friendPOI.csv --edge-path-friend input/hongzhi_friends.csv --lbsn hongzhi --listPOI input/location_hongzhi

for DATA in hongzhi NYC TKY Istanbul Jakarta KualaLampur
do
python src/main.py --edge-path input/${DATA}_friendPOI.csv --edge-path-friend input/${DATA}_friends.csv --lbsn ${DATA} --listPOI input/location_${DATA}
done

################################## ORIGRAPH C++ ######################################

mkdir results/cpp

for DATA in KualaLampur 
do
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat  --clean --num_epochs 1 --walk_length 6
done

################################## FOR PYTHON #########################################

for DATA in NYC
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done

for DATA in Jakarta
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in Istanbul
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in SaoPaulo
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in KualaLampur
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done

for DATA in TKY
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


python -u main.py --dataset_name KualaLampur --mode friend --input_type mat --clean --num_epochs 10




for DATA in hongzhi
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --clean --num_epochs 1 --lea
done

for DATA in NYC
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done

for DATA in Jakarta
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in Istanbul
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in SaoPaulo
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in KualaLampur
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done

for DATA in TKY
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done
      

######################## friend prediction #####################
    # copy POI
        # normal randomwalk

mode="persona_ori"

for data in NYC
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  
done

mode="persona_ori"

for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  > output/normalrw_copyPOI_${data}_friend
done



    # tach POI (WAIT...)
        # normal random walk

mode="persona_POI"

for data in NYC
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  
done

mode="persona_POI"


for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data}  > output/normalrw_sepPOI_${data}_friend
done

        # new random walk 

mode="persona_POI"

for data in NYC
do 
    python -u CMan.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --alpha 0.1
done

mode="persona_POI"

for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do 
    for alpha in 0.1 0.11 0.15 0.2
    do 
        python -u CMan.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --alpha ${alpha} > output/biasrw${alpha}_sepPOI_${data}_friend
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

for data in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_M.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_SM.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
done

# Just M
for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do
python src/hypergraph_embedding.py --data_path ../LBSN2Vec/dhne_graph/${data} --save_path ../LBSN2Vec/dhne_emb/${data} -s 16 16 16 -b 256
done

# eval_embedding

for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in line word2vec deepwalk
    do 
        python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > output/${model}_${data}_friend
        python -u eval_models.py --emb_path ${model}_emb/${data}_M.embeddings --dataset_name ${data} --model ${model} > output/${model}_${data}_M_friend
        python -u eval_models.py --emb_path ${model}_emb/${data}_SM.embeddings --dataset_name ${data} --model ${model} > output/${model}_${data}_SM_friend
    done

    model=dhne
    python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > output/${model}_${data}_friend

done 





######################## location prediction #####################
    # copy POI
        # normal randomwalk



    # tach POI (wait)
        # normal random walk

        # new random walk 

    # Original_version

    # other baselines