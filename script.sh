
################################## PERSONA ######################################

# persona: For one, we need to run splitter to create persona graph
for DATA in Istanbul Jakarta KualaLampur SaoPaulo TKY
do
    python src/main.py --edge-path input/${DATA}_friends.csv --lbsn ${DATA}
done 

# persona: then we just need to run LBSN with persona graphs
for DATA in Istanbul Jakarta KualaLampur SaoPaulo TKY NYC hongzhi
do
    python -u main.py --dataset_name ${DATA} --mode POI --input_type persona > results/POI_persona_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type persona > results/friend_persona_${DATA}
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
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
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
      
