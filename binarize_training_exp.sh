#/bin/bash
# finetuning by passing 4th argument srcdict tgtdict
exp_dir=$1
src_lang=$2
tgt_lang=$3
data_bin_dir=$4

# use cpu_count to get num_workers instead of setting it manually when running in different
# instances
num_workers=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`

data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

rm -rf $out_data_dir

fairseq-preprocess \
    --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $data_dir/train \
    --validpref $data_dir/dev \
    --testpref $data_dir/test \
    --destdir $out_data_dir \
    --workers $num_workers \
    --srcdict $data_bin_dir/dict.SRC.txt --tgtdict $data_bin_dir/dict.TGT.txt
