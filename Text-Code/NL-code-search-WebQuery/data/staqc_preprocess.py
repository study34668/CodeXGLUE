import io
import json
import pickle
import random
import tokenize
import tqdm

dataset_name = 'staqc'
title_data = pickle.load(open('qid_to_title.pickle', 'rb'))
code_data = pickle.load(open('qid_to_code.pickle', 'rb'))
train_num_negative = 3
valid_num_negative = 1
test_num_negative = 1


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def get_data_from_txt(txt_name, num_negative=0):
    pids = []
    with open(txt_name, 'r') as txt_f:
        for line in txt_f:
            line = line.strip()
            pids.append(int(line))

    data = []
    idx = 0
    for pid in tqdm.tqdm(pids, desc='{} generate tokens'.format(txt_name)):
        title = title_data[pid]
        code_f = io.StringIO(code_data[pid])
        tokens = tokenize.generate_tokens(code_f.readline)
        token_strings = []
        for token in tokens:
           token_strings.append(token.string)
        code = " ".join(token_strings)
        data.append({'idx': idx,
                     'doc': title,
                     'code': format_str(code),
                     'label': 1})
        idx += 1

    if num_negative <= 0:
        return data

    length = len(data)
    for idx_x in tqdm.tqdm(range(length), desc='{} generate negative'.format(txt_name)):
        random_selected = random.sample(data[:idx_x] + data[idx_x+1:length], num_negative)
        for i in range(num_negative):
            data.append({'idx': idx_x+length+1,
                         'doc': data[idx_x]['doc'],
                         'code': random_selected[i]['code'],
                         'label': 0})

    return data


def save_data_to_json(data, json_name):
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(data, f)


random.seed(1)
train_data = get_data_from_txt('{}_train.txt'.format(dataset_name), train_num_negative)
valid_data = get_data_from_txt('{}_valid.txt'.format(dataset_name), valid_num_negative)
test_data = get_data_from_txt('{}_test.txt'.format(dataset_name), test_num_negative)
save_data_to_json(train_data, 'train_{}_{}.json'.format(dataset_name, train_num_negative))
save_data_to_json(valid_data, 'valid_{}_{}.json'.format(dataset_name, valid_num_negative))
save_data_to_json(test_data, 'test_{}_{}.json'.format(dataset_name, test_num_negative))


# f = open('qid_to_title.pickle', 'rb')
# data = pickle.load(f)
# train_num = 80549
# valid_num = 3072
# test_num = 1673
# train_data = []
# valid_data = []
# test_data = []
# for pid in data:
#     if len(train_data) < train_num:
#         train_data.append(pid)
#     elif len(valid_data) < valid_num:
#         valid_data.append(pid)
#     else:
#         test_data.append(pid)
#
# train_f = open('staqc_train.txt', 'w')
# valid_f = open('staqc_valid.txt', 'w')
# test_f = open('staqc_test.txt', 'w')
# for pid in train_data:
#     train_f.write(str(pid) + '\n')
# for pid in valid_data:
#     valid_f.write(str(pid) + '\n')
# for pid in test_data:
#     test_f.write(str(pid) + '\n')
