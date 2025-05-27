#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
ref_fname=$6

tgt_langs=( "gu" "hi"  "kn" "mr")
for j in "${tgt_langs[@]}" 
do
    tgt_lang=$j
    infname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/test.$src_lang"
    outfname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/outtest.$tgt_lang.tok"
    ref_fname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/test.$tgt_lang.tok"
    # sacrebleu --tokenize none $ref_fname < $outfname -m chrf
    python calc-metrics.py --ref $ref_fname --predictions $outfname
    # python comet-eval.py  --src $infname --ref $ref_fname --predictions $outfname
    
    cat "$infname" >> "$input_txt"
    cat "$outfname" >> "$output_tok" 
    cat "$ref_fname" >> "$ref_tok" 
done
# python calc-metrics.py --ref $ref_tok --predictions $output_tok
# python comet-eval.py  --src $input_txt --ref $ref_tok --predictions $output_tok

echo `date`
echo "Translation completed"