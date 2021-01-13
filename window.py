"""
VAE visualization app definition utilizing tkinter.
"""
import numpy as np 
from PIL import ImageTk
from PIL import Image as Img ## conflict with tkinter.Image class
from tkinter import *



class VAE_window():
    """
    Simple class for visualizing the working of a Variational Autodecoder. \\
    Params: \\
    -vae          -> VAE model to visualize \\
    -latent_space -> size of the latent space, value 3 for provided models \\
    -precision    -> if between (0., 1.) defines the precision of latent space sliders,
    otherwise the number of increments to divide the sliders into
    """
    def __init__(self, vae, latent_space, precision=100):
        self.vae = vae
        self.latent_space = int(latent_space)
        if precision < 1. and precision != 0:
            self.scale = int(1./precision)
        else:
            self.scale = int(precision)
        self.win = None
        self.window_build()


    def window_build(self):
        """
        Builds the app window and initializes image.
        """
        self.win = Tk()
        self.sliders = [Scale(self.win, from_=-self.scale, to=self.scale, orient=HORIZONTAL, command=self.imgupdate) for i in range(self.latent_space)]
        self.labels = [Label(text='z ['+str(i)+']') for i in range(self.latent_space)]

        z = [0 for i in range(self.latent_space)]
        img = self.img_get(z)

        self.imglabel = Label(self.win, image=img)
        self.imglabel.configure(image=img)
        self.imglabel.image = img 

        self.extbtn = Button(self.win, text='EXIT', command=self.end)

        ## Grid arrangement
        for i, (label, slider) in enumerate(zip(self.labels, self.sliders)):
            label.grid(row=i, column=0)
            slider.grid(row=i, column=1)

        self.imglabel.grid(row=self.latent_space, column=0)
        self.extbtn.grid(row=self.latent_space+1, column=0)


    def end(self):
        """
        Releases window after app stops running.
        """
        self.win.destroy()
        self.win = None



    def run(self):
        """
        Mainloop function for the app.
        """
        if self.win is None:
            self.window_build()
        self.win.mainloop()
        if self.win is not None:
            self.win.destroy()
            self.win = None


    def imgupdate(self, _):
        """
        Updates predicted image after any of the sliders are moved.
        """
        z = [slider.get() / float(self.scale) for slider in self.sliders]
        img = self.img_get(z)

        self.imglabel.configure(image=img)
        self.imglabel.image = img

    def img_get(self, z):
        """
        Creates a ready image based on latent space. \\
        Params: \\ 
        -z -> latent space vector \\
        Returns: \\
        -tkinter PhotoImage object
        """

        def img_upscale(arr, upsize:int=6):
            """Shorthand for making the image bigger."""
            return np.repeat(np.repeat(arr, upsize, axis=0), upsize, axis=1)
        
        img = self.vae.dec.predict([z]) * 255
        img = np.squeeze(img).astype(np.uint8)
        img = img_upscale(img)
        img = Img.fromarray(img)
        img = ImageTk.PhotoImage(img)
        
        return img

    



    








