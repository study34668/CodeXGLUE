nohup python code/run_classifier.py \
	--model_type roberta \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--train_file train_staqc_2.json \
	--dev_file valid_staqc_0.json \
	--max_seq_length 200 \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 8 \
	--learning_rate 5e-5 \
	--num_train_epochs 10 \
	--gradient_accumulation_steps 1 \
	--warmup_steps 1000 \
	--save_steps 1000 \
	--evaluate_during_training \
	--data_dir ./data/ \
	--output_dir ./model_staqc \
	--encoder_name_or_path microsoft/codebert-base \
	--n_cpu 1 \
	--num_workers 0 \
	--seed 123456 2>&1 | tee log/train.log &

jobs -l > log/pid.log
