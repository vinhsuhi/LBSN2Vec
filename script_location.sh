
######################## location prediction #####################
    # copy POI (WAIT)
        # normal randomwalk

mode="persona_ori"

for data in Jakarta 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
done

mode="persona_ori"

for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} > output/normalrw_sepPOI_${data}_POI
done



    # tach POI (wait)
        # normal random walk

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
done

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --workers 32
done



        # new random walk 

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --p_n2v 1 -q_n2v 1 --workers 32
done

mode="persona_POI"

for data in NYC 
do 
    for alpha in 0.1 
    do
        python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
    done
done




    # Original_version


for data in NYC 
do 
    python -u baseline_POI.py --dataset_name ${data} 
done


for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do 
    python -u baseline_POI.py --dataset_name ${data} > output/original_${data}_POI
done


    # other baselines



#TODO: how to eval ????
# create_data --> embedding --> eval




for dataset in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in dhne deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model} --POI
    done
done




mode="persona_POI"

for data in TKY hongzhi KualaLampur Istanbul SaoPaulo NYC
do
    for qq in 0.4 0.5 0.6
    do
        for pp in 0.6 0.8 1
        do 
            python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --p_n2v ${pp} --q_n2v ${qq} --workers 32 --mobility_ratio 0.7 > output1/${data}_p${pp}_q${qq}_POI
        done 
    done
done


mode="persona_ori"

for p in 0.8
do
for q in 0.6 0.8 1
do
for lr in 0.0005 0.001
do
for dim in 256 300
do
for Kneg in 5 10
do
for data in Jakarta KualaLampur SaoPaulo NYC TKY hongzhi Istanbul
do
python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
--dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
--workers 12 --learning_rate ${lr} --K_neg ${Kneg} --mobility_ratio 0.7 --add_flag 0.9 > location_pred2/model2_${data}_${dim}_${lr}_${Kneg}_p${p}_q${q}m0.7
done
done
done
done
done
done



mode="persona_POI"

for p in 0.8
do
for q in 0.6 0.8 1
do
for lr in 0.0005 0.001
do
for dim in 256 300
do
for Kneg in 5 10
do
for data in Jakarta KualaLampur SaoPaulo
do
python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
--dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
--workers 12 --learning_rate ${lr} --K_neg ${Kneg} --mobility_ratio 0.7 --add_flag 0.9 > location_pred2/model2_${data}_${dim}_${lr}_${Kneg}_p${p}_q${q}m0.7
done
done
done
done
done
done




mode="persona_POI"

for data in SaoPaulo 
do
    for lr in 0.0005 
    do 
        for dim in 300
        do 
            for Kneg in 25 5
            do 
                python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
                --dataset_name ${data} --bias_randomwalk --p_n2v 0.8 --q_n2v 0.8 \
                --workers 32 --learning_rate ${lr} --K_neg ${Kneg} --workers 12 > sao_output/${dim}_${lr}_${Kneg}
            done
        done 
    done
done


