conda activate alma

<<<<<<< HEAD
# python src/trainer_cpo.py --base_model_dir /root/azurestorage/models/production/tmt-ysolar-20k --batch_size 4 --epoch_size 1 --expr_desc finetuning --expr_name tmt-cpo --gradient_accumulation_steps 8 --max_len 4096 --learning_rate 5e-5 --gradient_checkpointing True


torchrun --nproc-per-node 4 src/trainer_cpo.py --base_model_dir /root/azurestorage/models/production/tmt-ysolar-20k --batch_size 2 --epoch_size 1 --expr_desc finetuning --expr_name tmt-cpo --gradient_accumulation_steps 2 --max_len 4096 --learning_rate 5e-5 --gradient_checkpointing True
=======
python ../src/trainer_cpo.py --base_model_dir /root/azurestorage/models/production/tmt-ysolar-20k --batch_size 2 --epoch_size 1 --expr_desc finetuning --expr_name tmt-cpo --gradient_accumulation_steps 16 --max_len 4096 --learning_rate 5e-5 --gradient_checkpointing True
>>>>>>> 78696b827b3085b3b5a90dd239f7ea1ba424621e
