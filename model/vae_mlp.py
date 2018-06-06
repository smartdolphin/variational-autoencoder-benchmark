""" referenced by keras team(https://github.com/keras-team/keras)

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
"""
from keras.initializers import VarianceScaling
from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.losses import binary_crossentropy


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, latent_dim))
    return z_mean + K.exp(z_log_var * 0.5) * epsilon


def vae_mlp(original_dim: int, intermediate_dim: int, latent_dim: int, dropout_rate=0.1):
    input_shape = (original_dim, )

    # VAE model = encoder + decoder
    # build encoder model
    x = Input(shape=input_shape, name='encoder_input')
    # 1st hidden layer
    h0 = Dense(intermediate_dim, activation='elu', kernel_initializer=VarianceScaling())(x)
    h0 = Dropout(rate=dropout_rate)(h0)
    # 2nd hidden layer
    h1 = Dense(intermediate_dim, activation='tanh', kernel_initializer=VarianceScaling())(h0)
    h1 = Dropout(rate=dropout_rate)(h1)
    gaussian_params = Dense(latent_dim * 2, name='gaussian_params', kernel_initializer=VarianceScaling())(h1)
    # The mean parameter is unconstrained
    mean = Lambda(lambda param: param[:, :latent_dim])(gaussian_params)
    log_var = Lambda(lambda param: param[:, latent_dim:])(gaussian_params)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mean, log_var])

    # instantiate encoder model
    encoder = Model(x, [mean, log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # 1st hidden layer
    h1 = Dense(intermediate_dim, activation='tanh', kernel_initializer=VarianceScaling())(latent_inputs)
    h1 = Dropout(rate=dropout_rate)(h1)
    # 2nd hidden layer
    h2 = Dense(intermediate_dim, activation='elu', kernel_initializer=VarianceScaling())(h1)
    h2 = Dropout(rate=dropout_rate)(h2)
    y = Dense(original_dim, activation='sigmoid', kernel_initializer=VarianceScaling())(h2)

    # instantiate decoder model
    decoder = Model(latent_inputs, y, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    _, _, latent_z = encoder(x)
    y = decoder(latent_z)
    vae = Model(x, y, name='vae_mlp')

    # vae loss
    reconstruction_loss = -K.sum(x * K.log(y) + (1 - x) * K.log(1 - y), 1)
    kl_divergence = -0.5 * K.sum(1 + log_var - (K.square(mean) + K.exp(log_var)), axis=-1)
    vae_loss = K.mean(reconstruction_loss) + K.mean(kl_divergence)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae, encoder, decoder

