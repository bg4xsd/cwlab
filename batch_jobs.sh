#!/bin/bash
PATH=/home/quant/miniconda3/bin:/home/quant/miniconda3/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

echo "Init conda"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/quant/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/quant/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/quant/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH=".:/home/quant/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

echo "Activate nn39..."
conda activate nn39

echo "Change path"
cd /home/quant/myWork/quantlab/group

# Step 1
python main_pro.py  --len_low 1 --len_high 20 --batch_size 128 --lr 0.001 --epoch_start 0 --epoch_end 5000 --workers 8
# Step 2
python main_pro.py  --len_low 1 --len_high 20 --batch_size 128 --lr 0.0001 --epoch_start 5000 --epoch_end 10000 --workers 8