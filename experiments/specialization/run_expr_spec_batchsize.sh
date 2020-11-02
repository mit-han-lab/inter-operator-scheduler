for gbs in 1 32 128; do
    for ebs in 1 32 128; do
        python main.py --edir "batchsize" --ename gbs${gbs}_ebs${ebs} --device v100 --graph $gbs --bs $ebs
    done
done

