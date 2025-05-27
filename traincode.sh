CUDA_VISIBLE_DEVICES=1 CUDA_AVAILABLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 fairseq-train ../exp1/final_bin \
--max-source-positions=240 \
--max-target-positions=240 \
--max-update=5024260 \
--save-interval=1 \
--arch=transformer_4x \
--criterion=label_smoothed_cross_entropy \
--source-lang=SRC \
--lr-scheduler=inverse_sqrt \
--target-lang=TGT \
--label-smoothing=0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--dropout 0.2 \
--save-dir ../exp1/model_4x \
--keep-last-epochs 2 \
--patience 4 \
--skip-invalid-size-inputs-valid-test \
--fp16 \
--user-dir model_configs \
--update-freq=1 \
--distributed-world-size 1 \
--max-tokens 1024 \
--max-epoch 50 \
--scoring bleu \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--restore-file ../en-indic/model/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer \
--num-domains=3 \
--num-workers 0 \
 2>&1 | tee  ../exp1/log4x_residual_gate_wo_encoder_with_shared.txt



# /home/ganesh/lexgen/en-indic/model/checkpoint_best.pt \