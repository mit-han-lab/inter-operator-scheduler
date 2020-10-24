cur_device=$1
echo Run on $cur_device
schedules="k80 v100"
for schedule in $schedules; do
    python main.py --edir "device" --ename ${schedule}_on_${1} --device $1 --graph $schedule --bs 1
done
