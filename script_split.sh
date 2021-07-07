
######################## friend prediction #####################
    # copy POI
        # normal randomwalk

for data in NYC hongzhi TKY Jakarta KualaLampur SaoPaulo Istanbul
do
#    start=`date +%s.%N`
    python src/main.py --edge-path ./input/${data}_friendPOI.csv --lbsn ${data} --location-dict ./Suhi_output/location_dict_${data} --edge-path-friend ./input/${data}_friends.csv --listPOI ./input/location_${data}
#    end=`date +%s`
#    runtime=$((end-start))
done
