import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument('--output_test_file', default=None, type=str,
                        help="loc to store generated test file")
    parser.add_argument("--prepare_test_json", action='store_true',
                        help="gen test json for run_classifier")
    parser.add_argument("--output_answer", action='store_true',
                        help="get answer from prediction.txt")
    parser.add_argument('--query', default='', type=str, help="query text")
    args = parser.parse_args()

    if args.prepare_test_json:
        test_data_path = os.path.join(args.data_dir, args.test_file)
        if args.output_test_file:
            output_test_file_path = os.path.join(args.data_dir, args.output_test_file)
        else:
            output_test_file_path = os.path.join(args.data_dir, 'test_staqc_query.json')
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        for js in data:
            js['doc'] = args.query
        with open(output_test_file_path, 'w') as f:
            json.dump(data, f)

    if args.output_answer:
        print('output answer')


if __name__ == "__main__":
    main()
