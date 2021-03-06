import os, sys
from openpyxl import load_workbook
import numpy as np
import argh
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt


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


def plot_ds_stats(dir=DATA_DIR):
    dsa = load_ds(dir)

    # Image count distribution across topics
    f1 = plt.figure()
    topics = Counter(dsa['topics']).most_common()
    plt.plot([c for t, c in topics])
    plt.xlabel('Topics')
    plt.ylabel('#Images')
    plt.savefig(os.path.join(dir, 'img_topc_distribution.png'))

    # Income distribution
    f2 = plt.figure()
    plt.plot(sorted(dsa['income'], reverse=True))
    plt.xlabel('Images')
    plt.ylabel('Income')
    plt.savefig(os.path.join(dir, 'income_distribution.png'))


def most_common_topics(n=16, dir=DATA_DIR):
    dsa = load_ds(dir)
    topics = Counter(dsa['topics']).most_common(n)
    topics = [t.replace('/', '_') for t, _ in topics]
    return topics


def wget_topic_images(topic, save_dir, verbose=False):
    dsa = load_ds()
    urls = dsa[np.where(dsa['topics'] == topic)]['imageUrl']
    fn = topic.replace(' ', '_') + '_urls.txt'
    with open(fn, 'w') as f:
        f.write('\n'.join(urls))
    v = '-v' if verbose else ''
    os.system(f'wget -nc {v} -i {fn} -P {save_dir}')


def wget_most_common_topic_images(save_dir, n=16, verbose=False):
    topics = most_common_topics(n)
    for t in topics:
        print('########################## ' + t + ' ##########################')
        wget_topic_images(t, save_dir, verbose)


def create_income_quantile_images_dict(topic, img_dir=DATA_DIR, quantile=4, file_path=True, filter_existing_path=True):
    dsa = load_ds(img_dir)
    dsa_t = dsa[np.where(dsa['topics'] == topic)]
    q = int(len(dsa_t) / quantile)
    dsa_t.sort(order='income')
    dsa_cheap = dsa_t[:q]
    dsa_expensive = dsa_t[-q:]
    income_quantile_images_dict = {}

    def img_path(url):
        if file_path:
            return os.path.join(img_dir, os.path.basename(url))
        else:
            return url

    def path_filter(paths):
        if filter_existing_path:
            return list(filter(os.path.exists, paths))
        return paths

    income_quantile_images_dict['cheap'] = \
        pd.DataFrame(data={'path': path_filter([img_path(url) for url in dsa_cheap['imageUrl']])})
    income_quantile_images_dict['expensive'] = \
        pd.DataFrame(data={'path': path_filter([img_path(url) for url in dsa_expensive['imageUrl']])})
    return income_quantile_images_dict



if __name__ == '__main__':
    argh.dispatch_commands([wget_topic_images, wget_most_common_topic_images,
                            create_income_quantile_images_dict, plot_ds_stats])
