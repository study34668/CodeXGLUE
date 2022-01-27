import argparse
import json
import os
import pickle


def getLogit(item):
    return item['logit']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument('--output_test_file', default='query_staqc_doc.json', type=str,
                        help="loc to store generated test file")
    parser.add_argument("--prediction_file", default='evaluator/staqc_query_predictions.txt', type=str,
                        help='path to save predictions result, note to specify task name')
    parser.add_argument("--prepare_test_json", action='store_true',
                        help="gen test json for run_classifier")
    parser.add_argument("--output_answer", action='store_true',
                        help="get answer from prediction.txt")
    parser.add_argument('--query', default='', type=str, help="query text")
    args = parser.parse_args()

    if args.prepare_test_json:
        test_data_path = os.path.join(args.data_dir, args.test_file)
        output_test_file_path = os.path.join(args.data_dir, args.output_test_file)
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        for js in data:
            js['doc'] = args.query
        with open(output_test_file_path, 'w') as f:
            json.dump(data, f)

    if args.output_answer:
        test_data_path = os.path.join(args.data_dir, args.test_file)
        qid_to_code_data_path = os.path.join(args.data_dir, 'qid_to_code.pickle')
        code_data = pickle.load(open(qid_to_code_data_path, 'rb'))

        idx_pid_map = {}
        with open(test_data_path, 'r') as f:
            data = json.load(f)
            for js in data:
                idx_pid_map[js['idx']] = js['pid']

        list = []
        with open(args.prediction_file, 'r') as f:
            for line in f.readlines():
                pred = line.strip().split('\t')
                idx, logit = pred[0], float(pred[1])
                list.append({
                    'pid': idx_pid_map[idx],
                    'logit': logit
                })

        list.sort(reverse=True, key=getLogit)
        for i in range(10):
            print("******" + str(list[i]['logit']) + "******")
            print(code_data[list[i]['pid']])
            print("******************************")


if __name__ == "__main__":
    main()
