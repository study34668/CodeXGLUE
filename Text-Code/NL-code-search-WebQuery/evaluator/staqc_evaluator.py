import logging
import sys, json, os
import numpy as np
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for js in data:
            answers[js['idx']] = int(js['label'])
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predictions[line.split('\t')[0]] = int(line.split('\t')[2])
    return predictions


def calculate_scores(answers, predictions):
    y_trues, y_preds = [], []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores={}
    scores['Precision']=precision_score(y_trues, y_preds)
    scores['Recall']=recall_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    scores['Accuracy']=accuracy_score(y_trues, y_preds)
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for ClozeTest-maxmin dataset.')
    parser.add_argument('--answers_file', '-aw', help="filename of the labels on test set, in json format.")
    parser.add_argument('--predictions_file', '-pw', help="filename  of the leaderboard predictions on test set, in txt format.")
    args = parser.parse_args()

    answers = read_answers(args.answers_file)
    predictions = read_predictions(args.predictions_file)
    scores = calculate_scores(answers, predictions)
    print('NL-code-search-WebQuery predictions on test set:')
    print(scores)


if __name__ == '__main__':
    main()