sudo env PYTHONPATH=$PYTHONPATH $(which python) main.py --ename seq --event 0 > seq_0.txt
sudo env PYTHONPATH=$PYTHONPATH $(which python) main.py --ename ios --event 0 > ios_0.txt

python draw_curve.py --seq_log_file seq_0.txt --ios_log_file ios_0.txt