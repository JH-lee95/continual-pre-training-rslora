# CUDA_VISIBLE_DEVICES=0 python src/trainer_sft.py --base_model_dir beomi/OPEN-SOLAR-KO-10.7B --batch_size 4 --epoch_size 1 --expr_desc finetuning --expr_name tmt-kosolar --gradient_accumulation_steps 16 --max_len 4096 --gradient_checkpointing True

# python src/trainer_sft.py --base_model_dir yanolja/KoSOLAR-10.7B-v0.2 --batch_size 2 --epoch_size 1 --expr_desc finetuning --expr_name tmt-yanolja-kosolar --gradient_accumulation_steps 16 --max_len 4096 --learning_rate 5e-5 --gradient_checkpointing True

# torchrun --nproc-per-node 4 src/trainer_sft.py --base_model_dir yanolja/KoSOLAR-10.7B-v0.2 --batch_size 1 --epoch_size 1 --expr_desc finetuning --expr_name tmt-yanolja-kosolar --gradient_accumulation_steps 32 --max_len 4096 --learning_rate 5e-5



# torchrun --nproc-per-node 4 src/trainer_sft.py --base_model_dir yanolja/KoSOLAR-10.7B-v0.2 --batch_size 1 --epoch_size 1 --expr_desc finetuning --expr_name tmt-yanolja-kosolar-wsg --gradient_accumulation_steps 8 --max_len 4096 --learning_rate 5e-5





# torchrun --nproc-per-node 4 src/trainer_sft.py --base_model_dir yanolja/KoSOLAR-10.7B-v0.2 --batch_size 1 --epoch_size 1 --expr_desc "term_dict rearragned" --expr_name tmt-yanolja-kosolar-17k --gradient_accumulation_steps 8 --max_len 4096 --learning_rate 5e-5

# torchrun --nproc-per-node 4 src/trainer_sft.py --base_model_dir yanolja/EEVE-Korean-10.8B-v1.0 --batch_size 1 --epoch_size 1 --expr_desc "finetuning" --expr_name tmt-eeve-20k-sent2term-new-glossary-template --gradient_accumulation_steps 8 --max_len 4096 --learning_rate 1e-5



python src/trainer_sft.py --base_model_dir yanolja/EEVE-Korean-10.8B-v1.0 --batch_size 4 --epoch_size 1 --expr_desc "no glossary template whenre there is no glossary" --expr_name tmt-eeve-20k-sent2term-new-glossary-template --gradient_accumulation_steps 8 --max_len 4096 --learning_rate 1e-5 --gradient_checkpointing=True