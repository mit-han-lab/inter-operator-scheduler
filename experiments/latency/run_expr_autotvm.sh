ename="autotvm"
models="inception_v3 randwire nasnet squeezenet"
opt_types="tvm_tune"
cnt=1

for model in $models; do
	for bs in 1; do
		for opt_type in $opt_types; do
            for index in $(seq $cnt); do
                echo $model
                python main.py --ename $ename --device v100 --model $model --bs $bs --opt_type $opt_type --index $index
                echo end
            done
		done
	done
done

echo "Summaries: "
cat $(find outputs/. | grep "summary" | grep "$ename")
