#!/bin/bash
echo `date`
src_lang='en'
exp_dir=$1
test_dir=$2

rm -rf "output.tok"
rm -rf "ref_fname.tok"
rm -rf "input.txt"
rm -rf "score.txt"
output_tok="output.tok"
ref_tok="ref_fname.tok"
input_txt="input.txt"
tgt_langs=( "gu" "hi" "kn" "mr" "or" "ta")
# tgt_langs=( "gu" "hi"  "kn" "mr")
tgt_langs=("ta" )
for j in "${tgt_langs[@]}" 
do
    tgt_lang=$j
    infname="$exp_dir/$test_dir/all/$src_lang-$tgt_lang/test.$src_lang"
    outfname="$exp_dir/$test_dir/all/$src_lang-$tgt_lang/outtest.$tgt_lang"
    ref_fname="$exp_dir/$test_dir/all/$src_lang-$tgt_lang/test.$tgt_lang"
    # sacrebleu --tokenize none $ref_fname < $outfname -m chrf --chrf-word-order 1  --chrf-char-order 4
    python calc-metrics.py --ref $ref_fname --predictions $outfname
    # python copyinferencefile.py --ref $ref_fname --predictions $outfname
done

echo `date`
echo "Translation completed"
