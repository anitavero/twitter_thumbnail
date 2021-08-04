import os, sys
from openpyxl import load_workbook
import numpy as np
import argh
import pandas as pd
from collections import Counter


HOME_DIR = os.path.curdir
DATA_DIR = HOME_DIR + '/data'


def load_ds(dir=DATA_DIR):
    wb = load_workbook(filename=os.path.join(dir, 'DS images to Anita without country.xlsx'))
    ds = []
    rows = list(wb['Sheet1'].rows)
    for row in rows[1:]:
        ds.append(tuple(cell.value for cell in row[:-1]) + tuple([float(row[-1].value)]))
    types = [(k.value, 'U500') for k in rows[0][:-1]] + [(rows[0][-1].value, float)]
    dsa = np.array(ds, dtype=types)
    return dsa


def most_common_topics(n=16):
    dsa = load_ds()
    topics = Counter(dsa['topics']).most_common(n)
    topics = [t.replace('/', '_') for t, _ in topics]
    return topics


def wget_topic_images(topic, save_dir):
    dsa = load_ds()
    urls = dsa[np.where(dsa['topics'] == topic)]['imageUrl']
    fn = topic + 'urls.txt'
    with open(fn, 'w') as f:
        f.write('\n'.join(urls))
    os.system(f'wget -nc -v -i {fn} -P {save_dir}')


def wget_most_common_topic_images(save_dir, n=16):
    topics = most_common_topics(n)
    for t in topics:
        print('########################## ' + t + ' ##########################')
        wget_topic_images(t, save_dir)


def create_income_quantile_images_dict(topic, img_dir, quantile=4):
    dsa = load_ds(img_dir)
    dsa_t = dsa[np.where(dsa['topics'] == topic)]
    q = int(len(dsa_t) / quantile)
    dsa_t.sort(order='income')
    dsa_cheap = dsa_t[:q]
    dsa_expensive = dsa_t[-q:]
    income_quantile_images_dict = {}

    def img_path(url):
       return os.path.join(img_dir, os.path.basename(url))

    income_quantile_images_dict['cheap'] = \
        pd.DataFrame(data={'path': list(filter(os.path.exists, [img_path(url) for url in dsa_cheap['imageUrl']]))})
    income_quantile_images_dict['expensive'] = \
        pd.DataFrame(data={'path': list(filter(os.path.exists, [img_path(url) for url in dsa_expensive['imageUrl']]))})
    return income_quantile_images_dict



if __name__ == '__main__':
    argh.dispatch_commands([wget_topic_images, wget_most_common_topic_images])
