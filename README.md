# Variational Autoencoders
This was a coding exercise for me to familiarize myself with the working of Variational Autoencoder Neural Networks. I've created a Variational Autoencoder class (in `model.py`) and some utils to help visualize how it works. The models were build using keras and Tensorflow. 

# Quick Start
Before you run any of the code I recommend you read about the working of a VAE. If you'd like to see how it works for yourself, just run `main.py`. It will open a window where you can play around with the latent space values. If you want to see the working of the other model, just change the models name in the `main()` function.

# Included models
I have provided 2 pre-trained models. One was trained on the MNIST dataset and generates hand-written digits, the other was trained on FashionMNIST. Both of those datasets are freely available and even included with `keras`. 

# DISCLAIMER
You need tensorflow 2.4 to properly load the pre-trained models, otherwise tensorflow gives an error. This is due to the custom `Sampling` layer, which is the essence of a VAE. For some reason, tensorflow doesn't let you load custom layers from different versions, even though it was registered using tensorflow's decorator. You can install this version using `pip install tensorflow==2.4.x`.

# Other utils
I have also written several simple functions to help you see what goes on in the models. They're included in `visual.py`. You can make an image to see how the latent variables influence the outcome or even render a video. Keep in mind that calls to the model in the `video()` function come one after the other, which is not optimal and results in a very long runtime. You're free to optimize the function of you want.


