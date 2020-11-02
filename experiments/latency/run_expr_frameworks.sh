ename="framework"
models="inception_v3 randwire nasnet squeezenet"
opt_types="tf tf_xla taso tvm_cudnn trt"
cnt=1

for model in $models; do
    for bs in 1; do
        for opt_type in $opt_types; do
            for index in $(seq $cnt); do
                python main.py --ename $ename --device v100 --model $model --bs $bs --opt_type $opt_type --index $index
            done
        done
    done
done

echo "Summaries: "
cat $(find outputs/. | grep "summary" | grep "$ename")
