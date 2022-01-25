python evaluator/staqc_query.py \
	--data_dir ./data \
	--test_file test_staqc_0.json \
	--output_test_file test_staqc_query.json \
	--prepare_test_json \
	--query "how to get max value in python"

python code/run_classifier.py \
	--model_type roberta \
	--do_predict \
	--test_file test_staqc_query.json \
	--max_seq_length 200 \
	--per_gpu_eval_batch_size 8 \
	--data_dir ./data \
	--output_dir ./model_staqc/checkpoint-best-aver/ \
	--encoder_name_or_path microsoft/codebert-base \
	--prediction_file ./evaluator/staqc_query_predictions.txt

python evaluator/staqc_query.py \
	--data_dir ./data \
	--test_file test_staqc_0.json \
	--prediction_file evaluator/staqc_query_predictions.txt \
	--output_answer
