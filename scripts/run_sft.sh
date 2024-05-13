

# python src/run_training.py --base_model_dir maywell/EEVE-Korean-10.8B-v1.0-16k --train_batch_size 4 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve16k-data10k-split" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /azurestorage/data/translation_data/aligned_dataset/prepared_for_training/training_dataset_10k --lora True --gradient_checkpoint=True

torchrun --nproc-per-node 4  src/run_training.py --base_model_dir yanolja/EEVE-Korean-Instruct-10.8B-v1.0  --train_batch_size 4 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve-inst-data10k-split" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /data/training_dataset_10k --enable_lora True --optimizer pagedAdam8bit --gradient_checkpointing=True


# torchrun --nproc-per-node 4  src/run_training.py --base_model_dir yanolja/EEVE-Korean-10.8B-v1.0 --train_batch_size 4 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve-data10k-split" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /data/training_dataset_10k --enable_lora True --optimizer pagedAdam8bit --gradient_checkpointing=True

torchrun --nproc-per-node 4  src/run_training.py --base_model_dir yanolja/EEVE-Korean-10.8B-v1.0 --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve-data10k-split" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /data/training_dataset_10k --enable_galore True --enable_lora True



torchrun --nproc-per-node 4  src/run_training.py --base_model_dir maywell/EEVE-Korean-Instruct-10.8B-v1.0-32k --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve-inst32k-data7k" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_7k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing
