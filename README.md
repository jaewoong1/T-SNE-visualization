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
2. Set the **tile**, **method** and **save path**
3. run main.py 
```
$ python main.py --tile basic --method tsne --save result.png --path ./images
```
![tsne](https://user-images.githubusercontent.com/54341727/125231339-8cbdcf00-e315-11eb-9e52-84c8cf8793ae.png)

+ Added umap function and square tile mode
```
$ python main.py --tile square --method umap --save result.png --path ./images
```

![result](https://user-images.githubusercontent.com/54341727/125379356-593d7c00-e3cb-11eb-9abc-b827e2b8ffa0.png)
