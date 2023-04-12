import os

train_file_1 = open("./data/split/train_labeled_1_4.txt", 'r').readlines()
train_file_2 = open("./data/split/train_unlabeled_1_4.txt", 'r').readlines()

test_file = open('./data/split/test.txt', 'r').readlines()


def process_file(f):
    lines = []
    for line in f:
        line = line.rstrip().replace("\n", '')
        lines.append(line)
    return lines


train_file_1 = process_file(train_file_1)
train_file_2 = process_file(train_file_2)
test_file = process_file(test_file)
c = 0
for l in train_file_1:
    if l in test_file:
        print("file labeled: ", l)
        c += 1
print(c)

for l in train_file_2:
    if l in test_file:
        print("file unlabeled: ", l)
        c += 1

print(c)

for l in train_file_1:
    if l in train_file_2:
        print(l)
