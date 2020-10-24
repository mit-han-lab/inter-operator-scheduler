models="inception_v3 nasnet"
opt_types="dp_parallel_merge"

for model in $models; do
    for opt_type in $opt_types; do
        for r in 1 2 3; do
            for s in 3 8; do
                echo optimize $model using $opt_type with r $r s $s
                python main.py --device v100 --model $model --r $r --s $s --bs 1 --opt_type $opt_type --number 8 --repeat 4 #> /dev/null
            done
        done
    done
done


