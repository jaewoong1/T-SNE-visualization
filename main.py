import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from sklearn.manifold import TSNE
import os
import time
import umap
import argparse

def visualize_scatter_with_images(args, xs, ys, images, figsize=(45, 45)):

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for x0, y0, i in zip(xs, ys, images):
        img = OffsetImage(i)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim([[np.min(xs),np.min(ys)],[np.max(xs),np.max(ys)]])
    ax.autoscale()
    plt.savefig(args.save)

def load_data(args):

    ### set the path to the folder !!
    path = './images'

    imgs = []
    tsn = []
    images = os.listdir(path)
    for image in images:
        image = cv2.imread(os.path.join(path,image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        tsn.append(np.ravel(image, order='C'))
        imgs.append(image)

    if args.method == 'tsne':
        model = TSNE(learning_rate=1)
        proj = model.fit_transform(tsn)
    else:
        proj = umap.UMAP(n_neighbors=5, min_dist=0.3).fit_transform(tsn)

    xs = proj[:, 0]
    ys = proj[:, 1]

    return xs, ys, imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = time.time()
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                        help='Set the visualization method')
    parser.add_argument('--save', type=str, default='result.png', help='Save file name')

    args = parser.parse_args()
    print('Method :', args.method)
    xs, ys, imgs = load_data(args)

    visualize_scatter_with_images(args, xs, ys, images=imgs)
    print('Time : %f' % (time.time()-a))














