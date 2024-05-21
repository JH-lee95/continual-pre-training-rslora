torchrun --nproc-per-node 4  src/run_training.py --base_model_dir MLP-KTLim/llama-3-Korean-Bllossom-8B --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "llama3-blossom8b-data1.1k" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_1.1k --enable_lora --cache_dir /nvme0/models --gradient_checkpointing


torchrun --nproc-per-node 4  src/run_training.py --base_model_dir yanolja/EEVE-Korean-Instruct-10.8B-v1.0  --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "eeve-inst-data13k" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_13k --enable_lora --cache_dir /nvme0/models


torchrun --nproc-per-node 4  src/run_training.py --base_model_dir lcw99/llama-3-10b-it-kor-extented-chang --train_batch_size 1 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "llama3-10b-data11k" --max_seq_length 8192 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_11k --enable_lora --cache_dir /nvme0/models





python src/run_training.py --base_model_dir maywell/Yi-Ko-34B-Instruct --train_batch_size 2 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-34b-inst-data1.1k" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_1.1k --enable_lora --cache_dir /nvme0/models --device_map "auto"


python src/run_training.py --base_model_dir beomi/Yi-Ko-34B --train_batch_size 2 --expr_desc "finetuning" --expr_name "Term-MT" --run_name "yi-ko-34b-data1.1k" --max_seq_length 4096 --learning_rate 1e-5 --train_dataset_dir /nvme0/data/training_dataset_1.1k --enable_lora --cache_dir /nvme0/models --device_map "auto"
