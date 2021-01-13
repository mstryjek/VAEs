"""
Variational Autoencoder model definition and pre-trained models.
"""
import pkg_resources
pkg_resources.require('tensorflow==2.4.0') 
## Saved model tf version and loading tf version have to be same

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Reshape, Conv2DTranspose, Dropout
from keras import Input, layers

import numpy as np 




@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """
    Sampling (latent space) layer for Variational Autoencoders
    """

    def call(self, z_mean_logvar):
        """
        Latent space values based on latent space distribution.
        """
        z_mean, z_log_var = z_mean_logvar
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
      return super().get_config()





class VAE(Model):

    def __init__(self, imshape, latent_size, **kwargs):
        """
        Variational Autoencoder keras-based model class. \\
        Params: \\
        -imshape -> shape property of input and generated images \\
        -latent_size -> latent space vector length
        """
        super(VAE, self).__init__(**kwargs)
        self.imshape = imshape
        self.latent_size = latent_size
        self.enc = None
        self.dec = None


    def architecture_build(self):
        """
        Builds model architecture
        """

        def encoder_build():
            """
            Shorthand for encoder model architecture
            """
            enc_ins = Input(shape=self.imshape)
            x = Conv2D(32, 7, activation='relu', strides=2, padding='same')(enc_ins)
            x = Conv2D(64, 7, activation='relu', strides=2, padding='same')(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            z_mean = Dense(self.latent_size, name='z_mean')(x)
            z_log_var = Dense(self.latent_size, name='z_log_var')(x)
            z = Sampling()([z_mean, z_log_var])
            enc = Model(enc_ins, [z_mean, z_log_var, z], name='outer_encoder')
            return enc


        def decoder_build():
            """
            Shorthand for decoder model architecture
            """
            dec_ins = Input(shape=(self.latent_size, 1))
            x = Flatten()(dec_ins)
            x = Dense(self.imshape[0]*self.imshape[1]*128)(x)
            x = Reshape( [self.imshape[0], self.imshape[1], 128] )(x)
            x = Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
            x = Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
            dec_outs = Conv2DTranspose(self.imshape[2], 3, activation='relu', padding='same')(x)
            dec = Model(dec_ins, dec_outs, name='outer_decoder')
            return dec

        self.enc = encoder_build()
        self.dec = decoder_build()


    def train_step(self, data):
        """
        Custom train step function woth defined reconstruction, kulback-liebler and total loss functions.
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.enc(data)
            rec = self.dec(z)
            rec_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, rec))
            rec_loss *= self.imshape[0]*self.imshape[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -.5
            total_loss = rec_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'loss' : total_loss,
            'kl_loss' : kl_loss,
            'reconstruction_loss' : rec_loss}


    def compiled(self, optimizer):
        """
        Builds, compiles and returns working model.
        """
        if self.enc is None or self.dec is None: 
          self.architecture_build()
        self.compile(optimizer=optimizer)
        return self


    @classmethod
    def loads(cls, name, optimizer):
        """
        Dynamic model loading based on model names. Only works if model named correctly. Returns working model.
        """
        lsize = int(name.split('_')[-1])
        imshstr = name.split('_')[-2]
        imsh = tuple([int(n) for n in imshstr.split('x')])
        c = cls(imsh, lsize)
        c.enc = load_model(name + '_enc.h5')
        c.dec = load_model(name + '_dec.h5')
        return c.compiled(optimizer=optimizer)


    def saves(self, name):
        """
        Saves model components to .h5 files.
        """
        self.enc.save(name + '_enc.h5')
        self.dec.save(name + '_dec.h5')


def from_saved_models(mtype, model_dir=None):
    """
    Load one of the pre-trained models. \\
    Params: \\
    -mtype -> model type: \"mnist\" or \"fashion_mnist\" \\
    -model_dir -> (optional) directory where models are saved
    """
    if not model_dir:
        model_dir=''
    else:
        if model_dir[-1] != '/':
            model_dir = model_dir + '/'
    if mtype == 'mnist':
        return VAE.loads(model_dir + 'MNIST_28x28x1_3', 'adam')
    elif mtype == 'fashion_mnist':
        return VAE.loads(model_dir + 'FashionMNIST_28x28x1_3', 'adam')










