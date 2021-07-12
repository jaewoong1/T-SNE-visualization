# T-SNE-visualization-using-image-file

This code performs 2D t-sne visualization using image files.
It takes about 3 minutes for 1000 images.

## Requirements
* umap
* sklearn
* matplotlib
* opencv
* numpy

## Example

1. Put the image file you want to visualize in the folder ( ex : './images' )
2. run main.py 
```
$ python main.py --method tsne --save result.png
```
![tsne](https://user-images.githubusercontent.com/54341727/125231339-8cbdcf00-e315-11eb-9e52-84c8cf8793ae.png)

+ Added umap function
```
$ python main.py --method umap --save result.png
```

![result](https://user-images.githubusercontent.com/54341727/125233548-c264b700-e319-11eb-9b92-7d478c068b93.png)
