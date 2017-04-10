#!/usr/bash

cdssm_model='xxx/xxx.npz'
w2v_model='yyy/yyy.npz'

output_dir='../data/'

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
if [ ! -f $cdssm_model ]; then
    echo $cdssm_model" not exist."
elif [ ! -f $w2v_model ]; then
    echo $w2v_model" not exist."
else
    python py2cpp.py  $cdssm_model $w2v_model $output_dir
fi
