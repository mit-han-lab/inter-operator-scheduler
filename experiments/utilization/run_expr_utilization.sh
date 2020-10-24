sudo PATH=$PATH PYTHONPATH=$PYTHONPATH python main.py --ename seq > seq.txt
sudo PATH=$PATH PYTHONPATH=$PYTHONPATH python main.py --ename ios > ios.txt

python draw_curve.py --seq_log_file seq.txt --ios_log_file ios.txt


