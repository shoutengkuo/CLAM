"""Dataset CSVs for IENF with the third disease subtype iSFN.

labels.csv (used for dataset_csv/ienf_3cls.csv) has no iSFN rows; canonical_original_images.csv has all
439 images with their Category. This script takes Control / DM / FAP / iSFN from it (Unclassified is
dropped), keeps only slides with an image and an epidermis ROI, and checks that the 387 slides already in
ienf_3cls.csv keep their label.

  dataset_csv/ienf_isfn_4sub.csv   case_id, slide_id, label (Control / DM / FAP / iSFN), batch
                                   -> used for stratifying the splits
  dataset_csv/ienf_isfn_2cls.csv   case_id, slide_id, label (normal = Control, disease = DM + FAP + iSFN)
                                   -> used for training (task ienf_isfn_2cls)

Usage:
  python make_ienf_isfn_dataset_csv.py \\
      --data_root /data/datasets/neuroimages/20250214
"""
import argparse
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data_root', default='/data/datasets/neuroimages/20250214')
p.add_argument('--old_csv', default='dataset_csv/ienf_3cls.csv')
args = p.parse_args()
root = Path(args.data_root)

c = pd.read_csv(root / 'labels' / 'canonical_original_images.csv', encoding='utf-8-sig')
c['slide_id'] = c.SkinID + '_' + c.Side
c['batch'] = c.Canonical_Path.str.split('\\').str[1].map({'第一批': 'batch1', '第二批': 'batch2'})
have = {q.stem for q in (root / 'Image' / 'Original').glob('*.tif')} & {q.stem for q in (root / 'Epidermis_ROI').glob('*.roi')}
c = c[c.Category.isin(['Control', 'DM', 'FAP', 'iSFN']) & c.slide_id.isin(have)]
sub = pd.DataFrame({'case_id': c.SkinID, 'slide_id': c.slide_id, 'label': c.Category, 'batch': c.batch}
                   ).drop_duplicates('slide_id').sort_values(['case_id', 'slide_id'])
assert sub.batch.notna().all()
assert sub.groupby('case_id').label.nunique().max() == 1, 'a patient with two different labels'

old = pd.read_csv(args.old_csv).set_index('slide_id').label
m = sub.set_index('slide_id').label.reindex(old.index)
assert m.notna().all() and (m == old).all(), 'labels differ from the existing ienf_3cls.csv'

sub.to_csv('dataset_csv/ienf_isfn_4sub.csv', index=False)
two = sub[['case_id', 'slide_id']].assign(label=sub.label.map(lambda s: 'normal' if s == 'Control' else 'disease'))
two.to_csv('dataset_csv/ienf_isfn_2cls.csv', index=False)
print(f'{len(sub)} slides, {sub.case_id.nunique()} cases')
print(pd.crosstab(sub.label, sub.batch, margins=True).to_string())
print(two.label.value_counts().to_string())
