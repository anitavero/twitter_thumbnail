import os, sys
from openpyxl import load_workbook
import numpy as np


HOME_DIR = '/Users/anitavero/projects/twitter_thumbnail'
DATA_DIR = HOME_DIR + '/data'


def load_ds():
    wb = load_workbook(filename=os.path.join(DATA_DIR, 'DS images to Anita without country.xlsx'))
    ds = []
    rows = list(wb['Sheet1'].rows)
    for row in rows[1:]:
        ds.append(tuple(cell.value for cell in row[:-1]) + tuple([float(row[-1].value)]))
    types = [(k.value, 'U500') for k in rows[0][:-1]] + [(rows[0][-1].value, float)]
    dsa = np.array(ds, dtype=types)
    return dsa


def wget_topic_images(topic, save_dir):
    dsa = load_ds()
    urls = dsa[np.where(dsa['topics'] == topic)]['imageUrl']
    fn = topic + 'urls.txt'
    with open(fn, 'w') as f:
        f.write('\n'.join(urls))
    os.system(f'wget -v -i {fn} -P {save_dir}')
