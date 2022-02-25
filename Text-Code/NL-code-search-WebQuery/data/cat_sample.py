import pickle

title_data = pickle.load(open('qid_to_title.pickle', 'rb'))
code_data = pickle.load(open('qid_to_code.pickle', 'rb'))

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
title_list = []
code_list = []
for qid in title_data:
    title_list.append(title_data[qid])
    code_list.append(code_data[qid])
    if len(code_data[qid]) < 100:
        count1 += 1
    elif len(code_data[qid]) < 200:
        count2 += 1
    elif len(code_data[qid]) < 300:
        count3 += 1
    elif len(code_data[qid]) < 400:
        count4 += 1
    elif len(code_data[qid]) < 500:
        count5 += 1
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)

while True:
    i = int(input())
    print("********Title**********")
    print(title_list[i])
    print("********Code***********")
    print(code_list[i])
    print("********Length*********")
    print(len(code_list[i]))
    print("***********************")
