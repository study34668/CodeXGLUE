python code/clustering.py \
  --model_type roberta \
  --encoder_name_or_path microsoft/codebert-base \
	--max_seq_length 200 \
	--per_gpu_batch_size 8 \
	--data_dir ./data \
	--output_dir ./model_staqc \
	--pred_model_dir checkpoint-best-aver \
	--test_file query_staqc_0.json \
	--output_test_file query_staqc_doc.json \
	--prepare_test_json \
	--query "your question"

python code/run_classifier.py \
	--model_type roberta \
	--do_predict \
	--test_file query_staqc_doc.json \
	--max_seq_length 200 \
	--per_gpu_eval_batch_size 8 \
	--data_dir ./data \
	--output_dir ./model_staqc/checkpoint-best-aver/ \
	--encoder_name_or_path microsoft/codebert-base \
	--prediction_file ./evaluator/staqc_query_predictions.txt

python code/clustering.py \
	--data_dir ./data \
	--test_file query_staqc_0.json \
	--prediction_file evaluator/staqc_query_predictions.txt \
	--output_answer
