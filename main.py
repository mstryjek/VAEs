from model import *
from window import VAE_window
from visual import video


def main():
    """
    Runs an app that visualizes the how changing the hidden variables of a VAE impacts the created image. \\
    If you want to try a different model, replace the model loading function with your own or change its argument to either
    \"mnist\" or \"fashion_mnist\".
    """
    vae = from_saved_models('mnist', 'models') ## Model loading function
    app = VAE_window(vae, 3)
    app.run()




if __name__ == '__main__':
    main()




