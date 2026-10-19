"""
Official CAMELYON16 / CAMELYON17 train/test splits for Camelyon+.

  train: CAMELYON16 training slides (random_slide_*) + CAMELYON17 training patients (patient_000 - patient_099)
  test:  CAMELYON16 test slides (test_*)            + CAMELYON17 test patients (patient_100 - patient_199)

The official splits have no validation set, so val_frac of the training cases is held out for
validation / early stopping (case-level, stratified by label). Each fold uses a disjoint val set;
the test set is identical across folds.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Creating official Camelyon+ splits')
parser.add_argument('--csv_path', type=str, default='dataset_csv/camelyon_plus_2cls.csv')
parser.add_argument('--split_dir', type=str, default='splits/camelyon_plus_2cls_official')
parser.add_argument('--k', type=int, default=5, help='number of folds (different val sets, same test set)')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

def is_official_test(slide_id):
    if slide_id.startswith('test_'):
        return True
    m = re.match(r'patient_(\d+)_node_\d+', slide_id)
    return m is not None and int(m.group(1)) >= 100

df['official_test'] = df['slide_id'].map(is_official_test)
train_df, test_df = df[~df['official_test']], df[df['official_test']]

# case-level label: a case (patient) is positive if any of its slides is positive
case_label = train_df.groupby('case_id')['label'].agg(lambda x: 'tumor' if (x == 'tumor').any() else 'negative')
rng = np.random.RandomState(args.seed)
n_chunks = int(round(1 / args.val_frac))
assert args.k <= n_chunks, 'k must be <= 1 / val_frac so that val sets are disjoint'
chunks = [[] for _ in range(n_chunks)]
for label in sorted(case_label.unique()):
    cases = case_label[case_label == label].index.values.copy()
    rng.shuffle(cases)
    for j, part in enumerate(np.array_split(cases, n_chunks)):
        chunks[j].extend(part.tolist())

os.makedirs(args.split_dir, exist_ok=True)
for i in range(args.k):
    val_cases = set(chunks[i])
    val_slides = train_df[train_df['case_id'].isin(val_cases)]['slide_id']
    tr_slides = train_df[~train_df['case_id'].isin(val_cases)]['slide_id']
    splits = pd.concat([tr_slides.reset_index(drop=True).rename('train'),
                        val_slides.reset_index(drop=True).rename('val'),
                        test_df['slide_id'].reset_index(drop=True).rename('test')], axis=1)
    splits.to_csv(os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))

    descriptor = pd.DataFrame({name: df[df['slide_id'].isin(s)]['label'].value_counts()
                               for name, s in [('train', tr_slides), ('val', val_slides), ('test', test_df['slide_id'])]})
    descriptor.to_csv(os.path.join(args.split_dir, 'splits_{}_descriptor.csv'.format(i)))
    print('fold {}:\n{}\n'.format(i, descriptor))
