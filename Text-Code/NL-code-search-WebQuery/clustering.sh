python code/clustering.py \
	--model_type roberta \
	--clustering_file query_staqc_0.json \
	--max_seq_length 200 \
	--per_gpu_batch_size 8 \
	--data_dir ./data \
	--output_dir ./model_staqc \
	--encoder_name_or_path microsoft/codebert-base \
	--pred_model_dir checkpoint-best-aver \
	--do_clustering 2>&1 | tee log/clustering.log &

jobs -l > log/pid.log
