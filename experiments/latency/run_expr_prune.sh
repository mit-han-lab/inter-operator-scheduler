ename="prune"
models="inception_v3 nasnet"
opt_types="dp_parallel_merge"
cnt=1

for model in $models; do
    for opt_type in $opt_types; do
        for r in 1 2 3; do
            for s in 3 8; do
                for index in $(seq $cnt); do
                    python main.py --ename $ename --device v100 --model $model --r $r --s $s --bs 1 --opt_type $opt_type --index $index --number 8 --repeat 4
                done
            done
        done
    done
done


