import os
import math
import random

random.seed(1)

train = 0.5
val = 0.0
test = 0.5

root = r"D:\workspace\datasets\MMFV-25th"
train_file = 'outputs/train.txt'
test_file = 'outputs/test.txt'

all_subjects = []

for idx, subject in enumerate(os.listdir(root)):
    subject_path = os.path.join(root, subject)

    # Check path is a directory
    if not os.path.isdir(subject_path):
        continue

    # Skip single session subjects
    sessions = os.listdir(subject_path)
    if len(sessions) == 2:
        all_subjects.append(subject)

all_subjects = sorted(all_subjects, key=lambda x: int(x))

num_train = math.ceil(len(all_subjects) * train)
num_test = len(all_subjects) - num_train

train_subjects = random.sample(all_subjects, k=num_train)
train_subjects = sorted(train_subjects, key=lambda x: int(x))
test_subjects = [i for i in all_subjects if i not in train_subjects]
test_subjects = sorted(test_subjects, key=lambda x: int(x))

print(f'Num train : {len(train_subjects)}')
print(f'Num test  : {len(test_subjects)}')

with open(train_file, 'w') as f:
    for i in train_subjects:
        f.write(f'{i}\n')

with open(test_file, 'w') as f:
    for i in test_subjects:
        f.write(f'{i}\n')

with open(train_file) as f:
    content = f.readlines()
    content = [i.strip() for i in content]
    print(content)
