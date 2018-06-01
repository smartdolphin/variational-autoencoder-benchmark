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
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + z_log_var * epsilon


def vae_mlp(original_dim: int, intermediate_dim: int, latent_dim: int, dropout_rate=0.1):
    input_shape = (original_dim, )

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    # 1st hidden layer
    x = Dense(intermediate_dim, activation='elu', kernel_initializer=VarianceScaling())(inputs)
    x = Dropout(rate=dropout_rate)(x)
    # 2nd hidden layer
    x = Dense(intermediate_dim, activation='tanh', kernel_initializer=VarianceScaling())(x)
    x = Dropout(rate=dropout_rate)(x)
    gaussian_params = Dense(latent_dim * 2, name='gaussian_params', kernel_initializer=VarianceScaling())(x)
    # The mean parameter is unconstrained
    mean = Lambda(lambda param: param[:, :latent_dim])(gaussian_params)
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    stddev = Lambda(lambda param: 1e-6 + K.softplus(param[:, latent_dim:]))(gaussian_params)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mean, stddev])

    # instantiate encoder model
    encoder = Model(inputs, [mean, stddev, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # 1st hidden layer
    x = Dense(intermediate_dim, activation='tanh', kernel_initializer=VarianceScaling())(latent_inputs)
    x = Dropout(rate=dropout_rate)(x)
    # 2nd hidden layer
    x = Dense(intermediate_dim, activation='elu', kernel_initializer=VarianceScaling())(x)
    x = Dropout(rate=dropout_rate)(x)
    outputs = Dense(original_dim, activation='sigmoid', kernel_initializer=VarianceScaling())(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    y = decoder(encoder(inputs)[2])
    y = Lambda(lambda out: K.clip(out, 1e-8, 1 - 1e-8))(y)
    vae = Model(inputs, y, name='vae_mlp')

    # vae loss
    marginal_likelihood = K.sum(inputs * K.log(y) + (1 - inputs) * K.log(1 - y), 1)
    kl_divergence = K.mean(0.5 * K.sum(K.sqrt(mean) + K.sqrt(stddev) - K.log(1e-8 + K.sqrt(stddev)) - 1, 1))
    marginal_likelihood = K.mean(marginal_likelihood)
    elbo = marginal_likelihood - kl_divergence
    vae_loss = -elbo

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae, encoder, decoder

