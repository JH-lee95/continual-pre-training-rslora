

torchrun --nproc-per-node 4  src/run_training.py --base_model_dir maywell/Yi-Ko-34B-Instruct --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-inst-34b-stage1-sharegpt8k-even" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_sharegpt_8k_even --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 16 --lora_alpha 16

torchrun --nproc-per-node 4  src/run_training.py --base_model_dir maywell/Yi-Ko-34B-Instruct --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-inst-34b-datak7k" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_new_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 16 --lora_alpha 16




torchrun --nproc-per-node 4  src/run_training.py --base_model_dir /nvme0/models/yi-ko-34b-stage1-sharegpt8k-even --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-34b-stage2-sharegpt8k-even-data7k" --max_seq_length 4096 --learning_rate 5e-6 --train_dataset_dir /nvme0/data/training_dataset_new_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 4 --lora_alpha 4

torchrun --nproc-per-node 4  src/run_training.py --base_model_dir beomi/Yi-Ko-34B --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-34b-data7k" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_new_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 4 --lora_alpha 4


torchrun --nproc-per-node 4  src/run_training.py --base_model_dir lightblue/suzume-llama-3-8B-multilingual-orpo-borda-half --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "suzume-llama3-data7k" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_new_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 16 --lora_alpha 16



torchrun --nproc-per-node 4 src/run_training.py --base_model_dir CohereForAI/aya-23-35B --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "tmt-35b-data7k" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_new_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing --lora_rank 4 --lora_alpha 4

