#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
ref_fname=$6


# tgt_langs=( "as" "bn" "pa" "ml" "te" "gu" "hi" "kn" "mr" "or" "ta")
# tgt_langs=( "gu" "hi"  "kn" "mr" "or" "ta")
tgt_langs=( "gu" "hi"  "kn" "mr")
# tgt_langs=( "bn" "pa" "ml" "te" )
for j in "${tgt_langs[@]}" 
do
    tgt_lang=$j
    infname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/test.$src_lang"
    outfname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/outtest.$tgt_lang"
    ref_fname="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/test.$tgt_lang"
    domfile="$exp_dir/devtestexp2/all/$src_lang-$tgt_lang/test.dom"
    domList="test-domain.txt"
    rm -rf "$domList"
    touch $domList
    cat "$domfile" >> "$domList"
    SRC_PREFIX='SRC'
    TGT_PREFIX='TGT'

    #`dirname $0`/env.sh
    SUBWORD_NMT_DIR='subword-nmt'
    model_dir=$exp_dir/model_4x
    data_bin_dir=$exp_dir/final_bin

    ### normalization and script conversion

    echo "Applying normalization and script conversion"
    input_size=`python scripts/preprocess_translate.py $infname $outfname.norm $src_lang true`
    echo "Number of sentences in input: $input_size"

    ### apply BPE to input file

    echo "Applying BPE"
    python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
        -c $exp_dir/vocab/bpe_codes.32k.${SRC_PREFIX}\
        --vocabulary $exp_dir/vocab/vocab.$SRC_PREFIX \
        --vocabulary-threshold 5 \
        < $outfname.norm \
        > $outfname._bpe

    # not needed for joint training
    # echo "Adding language tags"
    python scripts/add_tags_translate.py $outfname._bpe $outfname.bpe $src_lang $tgt_lang

    ### run decoder

    echo "Decoding"

    src_input_bpe_fname=$outfname.bpe
    tgt_output_fname=$outfname
    CUDA_VISIBLE_DEVICES=0 fairseq-interactive  $data_bin_dir \
        -s $SRC_PREFIX -t $TGT_PREFIX \
        --distributed-world-size 1  \
        --path $model_dir/checkpoint_best.pt \
        --batch-size 64  --buffer-size 2500 --beam 5  --remove-bpe \
        --skip-invalid-size-inputs-valid-test \
        --user-dir model_configs \
        --input $src_input_bpe_fname  >  $tgt_output_fname.log 2>&1


    echo "Extracting translations, script conversion and detokenization"
    # this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
    python scripts/postprocess_translate.py $tgt_output_fname.log $tgt_output_fname $input_size $tgt_lang true

    # This block is now moved to compute_bleu.sh for release with more documentation.
    if [ $src_lang == 'en' ]; then
        # indicnlp tokenize the output files before evaluation
        input_size=`python scripts/preprocess_translate.py $ref_fname $ref_fname.tok $tgt_lang`
        input_size=`python scripts/preprocess_translate.py $tgt_output_fname $tgt_output_fname.tok $tgt_lang`
        sacrebleu --tokenize none $ref_fname.tok < $tgt_output_fname.tok -m chrf
    else
        # indic to en models
        sacrebleu $ref_fname < $tgt_output_fname
    fi
done
rm -rf "output.tok"
rm -rf "ref_fname.tok"
rm -rf "input.txt"
rm -rf "score.txt"
output_tok="output.tok"
ref_tok="ref_fname.tok"
input_txt="input.txt"
# tgt_langs=( "gu" "hi" "kn" "mr" "or" "ta")
tgt_langs=( "gu" "hi"  "kn" "mr")
# tgt_langs=( "kn" "mr" "or" )
# tgt_langs=( "hi" )
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
