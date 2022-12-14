 nohup  /bin/python./main.py > test.log 2>&1 &   

python main_pro.py  --len_low 12 --len_high 24 --batch_size 64 --lr 0.001 --epoch_start 8000 --epoch_end 10000 --workers 6