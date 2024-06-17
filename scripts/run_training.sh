torchrun --nproc-per-node 4  src/run_training.py --base_model_dir Qwen/Qwen2-7B --train_batch_size 2 --expr_desc "finetuning" --expr_name "cpt" --run_name "qwen2-7b-koen-cpt" --max_seq_length 8192 --learning_rate 1e-4 --enable_lora --cache_dir /nvme0/models --lora_rank 128 --lora_alpha 128 --gradient_checkpointing --eval=True


python src/run_training.py --base_model_dir Qwen/Qwen2-7B --train_batch_size 3 --expr_desc "finetuning" --expr_name "cpt" --run_name "qwen2-7b-koen-cpt" --max_seq_length 8192 --learning_rate 5e-6 --enable_lora --cache_dir /nvme0/models --lora_rank 256 --lora_alpha 256 --gradient_checkpointing --eval=True --logging_steps 10
