nohup python code/run_classifier.py \
	--model_type roberta \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--train_file train_staqc_7.json \
	--dev_file valid_staqc_0.json \
	--max_seq_length 200 \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 8 \
	--learning_rate 1e-5 \
	--num_train_epochs 2 \
	--gradient_accumulation_steps 1 \
	--warmup_steps 100 \
	--save_steps 2000 \
	--evaluate_during_training \
	--data_dir ./data/ \
	--output_dir ./model_staqc \
	--encoder_name_or_path microsoft/codebert-base \
	--n_cpu 1 \
	--num_workers 0 \
	--seed 2077 2>&1 | tee log/train.log &

jobs -l > log/pid.log
