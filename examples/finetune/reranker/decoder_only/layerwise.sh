export WANDB_MODE=disabled

train_data="\
    ../example_data/normal/examples.jsonl "

# set large epochs and small batch size for testing
num_train_epochs=4
per_device_train_batch_size=2
gradient_accumulation_steps=1
train_group_size=8

# set num_gpus to 2 for testing
num_gpus=2

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path BAAI/bge-reranker-base \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
"

training_args="\
    --output_dir ./test_encoder_only_base_bge-reranker-base \
    --overwrite_output_dir \
    --learning_rate 6e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd

torchrun --nproc_per_node 8 \
-m FlagEmbedding.finetune.reranker.decoder_only.layerwise \
--output_dir ./test \
--model_name_or_path /share/chaofan/models/minicpm-2b-fp32-dpo \
--train_data /share/chaofan/dataset/mteb_data_new_score/en/fiqa.jsonl \
--cache_dir /share/shared_models \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--max_steps 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--dataloader_drop_last True \
--gradient_checkpointing \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_total_limit 50 \
--fp16 \
--dataloader_drop_last True \
--weight_decay 0.01 \
--cache_path ./data \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn True \
--target_modules q_proj k_proj v_proj o_proj linear_head \
--save_merged_lora_model True \
--model_type decoder \
--deepspeed /share/chaofan/code/stage/stage1.json \
--model_type from_raw_model \
--start_layer 8 \
--head_multi True \
--head_type simple \
--trust_remote_code True
