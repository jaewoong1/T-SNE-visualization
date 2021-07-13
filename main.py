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
import rasterfairy
from PIL import Image
import math
from matplotlib.pyplot import imshow
import argparse

def visualization(args, proj, nx, ny, images, figsize=(45, 45)):

    if args.tile == 'square':
        grid_assignment = rasterfairy.transformPointCloud2D(proj, target=(nx, ny))

        tile_width = 100
        tile_height = 100

        full_width = tile_width * nx
        full_height = tile_height * ny
        aspect_ratio = float(tile_width) / tile_height

        grid_image = Image.new('RGB', (full_width, full_height))

        for img, grid_pos in zip(images, grid_assignment[0]):
            idx_x, idx_y = grid_pos
            x, y = tile_width * idx_x, tile_height * idx_y
            tile = Image.open(os.path.join(args.path, img))
            tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
            if (tile_ar > aspect_ratio):
                margin = 0.5 * (tile.width - aspect_ratio * tile.height)
                tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
            else:
                margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
                tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
            tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
            grid_image.paste(tile, (int(x), int(y)))

        grid_image = grid_image.rotate(270)
        matplotlib.pyplot.figure(figsize=(16, 16), dpi=500)
        plt.imshow(grid_image)
        plt.savefig(args.save)

    elif args.tile == 'basic':
        xs = proj[:, 0]
        ys = proj[:, 1]
        fig, ax = plt.subplots(figsize=figsize)
        artists = []
        for x0, y0, i in zip(xs, ys, images):
            image = cv2.imread(os.path.join(args.path, i))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (100, 100))
            img = OffsetImage(image)
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim([[np.min(xs),np.min(ys)],[np.max(xs),np.max(ys)]])
        ax.autoscale()
        plt.savefig(args.save)

def load_data(args):

    tsn = []
    images = os.listdir(args.path)
    ny = nx = math.ceil(np.sqrt(len(images)))
    for image in images:
        image = cv2.imread(os.path.join(args.path, image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        tsn.append(np.ravel(image, order='C'))

    if args.method == 'tsne':
        model = TSNE(learning_rate=1)
        proj = model.fit_transform(tsn)
    else:
        proj = umap.UMAP(n_neighbors=5, min_dist=0.3).fit_transform(tsn)

    return proj, images, nx, ny

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = time.time()
    parser.add_argument('--path', type=str, default='./images', help='Set the path of image folder')
    parser.add_argument('--tile', type=str, default='square', choices=['square', 'basic'],
                        help='Set the visualization method')
    parser.add_argument('--method', type=str, default='umap', choices=['tsne', 'umap'],
                        help='Set the visualization method')
    parser.add_argument('--save', type=str, default='result.png', help='Save file name')

    args = parser.parse_args()
    print('Method :', args.method)
    proj, imgs, nx, ny = load_data(args)

    visualization(args, proj, nx, ny, images=imgs)
    print('Time : %f' % (time.time()-a))














