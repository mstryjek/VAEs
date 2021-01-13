"""
VAE visualization resources.
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os




def indir(directory):
    """
    Decorator for running a function in a specific directory, remaining in the original directory 
    after the function has returned, in order not to pollute working directory. \\
    Used here for video() which creates many files.
    """
    def inside(fn):

        def wrapper(*args, **kwargs):
            original_dir = os.getcwd()
            try:
                os.chdir(directory)
            except:
                os.mkdir(directory)
                os.chdir(directory)
            fn(*args, **kwargs)
            os.chdir(original_dir)

        return wrapper

    return inside



def latent_space_grid(vae, wshift, imgsize=28, n=30, figsize=15, useaxes=True):
    """
    Plots a grid of latent space variables for a VAE with latent space size 3. \\
    Params:\\
    -vae -> model to use. Should have a .dec member \\
    -wshift -> w (third latent space argument) value to use \\
    -imgsize -> size of the square images in pixels \\
    -n -> number of images (per side) to show \\
    -figsize -> figure size, same as in matplotlib \\
    -useaxes -> whether to show values on axes or return a plain image. \\
    Returns:\\
    Path to saved .png image
    """
    scale = 1.0
    figure = np.zeros((imgsize * n, imgsize * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi, wshift]])
            x_decoded = vae.dec.predict(z_sample)
            digit = x_decoded[0].reshape(imgsize, imgsize)
            figure[
                i * imgsize : (i + 1) * imgsize,
                j * imgsize : (j + 1) * imgsize,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = imgsize // 2
    end_range = n * imgsize + start_range
    pixel_range = np.arange(start_range, end_range, imgsize)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    if useaxes:
        plt.xlabel("z [0]")
        plt.ylabel("z [1]")
        plt.title(f'z [2] = {wshift}')
    else:
        plt.axis('off')
    plt.imshow(figure, cmap="Greys_r")
    imgname = 'w_' + str(wshift) + '.png'
    plt.savefig(imgname)
    plt.close('all')
    return imgname



@indir('video')
def video(vae, vidpath, imgsize=28, frames=192, fps=24, n=30, fgisize=15, useaxes=True):
    """
    Creates and saves a video visualizing a latent space grid across the third latnet space dimension. \\
    Params: \\
    -vae -> model to visualize \\
    -vidpath -> path where to save the video \\
    -imgsize -> size of every predited image in px \\
    -frames -> length of the video in frames \\
    -fps -> frames per second. Standard value is 24 \\
    -n -> number of images to show per size \\
    -figsize -> standard figsize param like in matplotlib \\
    -useaxes -> whether to show axes with specific latent space values
    """
    increments = np.linspace(-1, 1, frames)
    images = [ ]
    i = 0
    for increment in increments:
        img = latent_space_grid(vae, increment, imgsize, n, fgisize, useaxes)
        images.append(img)
        i += 1
        print(f'Rendered frame {i}')
    imgs = [cv2.imread(img) for img in images]
    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(vidpath, fourcc, fps, (width, height))
    for img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()






