EXPERIMENT_DIR="/home/challenger/hyperinverter/experiment"
W_ENCODER_PATH="/home/challenger/hyperinverter/pretrained_models/w_encoder_e4e_ffhq_encode.pt"
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/train.py \
--dataset_type=ffhq_encode \
--encoder_type=LayerWiseEncoder \
--w_encoder_path="$W_ENCODER_PATH" \
--output_size=1024 \
--exp_dir="$EXPERIMENT_DIR" \
--batch_size=2 \
--batch_size_used_with_adv_loss=4 \
--workers=4 \
--val_interval=1000 \
--save_interval=5000 \
--encoder_optim_name=adam \
--discriminator_optim_name=adam \
--encoder_learning_rate=1e-4 \
--discriminator_learning_rate=1e-4 \
--hyper_lpips_lambda=0.8 \
--hyper_l2_lambda=1.0 \
--hyper_id_lambda=0.1 \
--hyper_adv_lambda=0.005 \
--hyper_d_reg_every=16 \
--hyper_d_r1_gamma=10.0 \
--step_to_add_adversarial_loss=200000 \
--target_shape_name=conv_without_bias  \
--max_steps=500000 \
--hidden_dim=128 \
--num_cold_steps=20000 \
--save_checkpoint_for_resuming_training \
--use_wandb