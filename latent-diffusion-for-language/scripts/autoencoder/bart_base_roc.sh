export TOKENIZERS_PARALLELISM=false
python train_latent_model.py --dataset_name librispeech --enc_dec_model facebook/bart-base --learning_rate 1e-4 --lr_warmup_steps 1000 --train_batch_size 356 --num_encoder_latents 32 --dim_ae 64 --num_decoder_latents 32  --eval_every 1000 --num_layers 1 --wandb_name bart-roc-l2norm-test-32-64 --l2_normalize_latent --mixed_precision fp16

