import sys
import csv
args = sys.argv

with open(args[1], mode='r') as pred:
    pred_list = pred.read().splitlines()

with open(args[2], mode='r') as gt:
    reader = csv.reader(gt)
    # gt_list = [rows[5] for rows in reader][1:]
    gt_list = gt.read().splitlines()

assert len(pred_list) == len(gt_list)

total_count = 0
correct_count = 0
for pred, gt in zip(pred_list, gt_list):
    if int(pred) == int(gt):
        correct_count += 1
    total_count += 1

accuracy = (correct_count / total_count) * 100
print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))
