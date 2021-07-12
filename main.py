import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from sklearn.manifold import TSNE
import os
import time

def visualize_scatter_with_images(xs, ys, images, figsize=(45, 45)):

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for x0, y0, i in zip(xs, ys, images):
        img = OffsetImage(i)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim([[np.min(xs),np.min(ys)],[np.max(xs),np.max(ys)]])
    ax.autoscale()
    plt.savefig('tsne.png')

def load_data():

    imgs = []
    tsn = []
    images = os.listdir('../image')
    for image in images:
        image = cv2.imread('../image/'+image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        tsn.append(np.ravel(image, order='C'))
        imgs.append(image)

    model = TSNE(learning_rate=1)
    transformed = model.fit_transform(tsn)
    xs = transformed[:, 0]
    ys = transformed[:, 1]

    return xs, ys, imgs

if __name__ == '__main__':
    a = time.time()
    xs, ys, imgs = load_data()

    visualize_scatter_with_images(xs, ys, images=imgs)
    print(time.time()-a)














