import argparse
from keras.utils import plot_model
from model.vae_mlp import vae_mlp
from model.vae_conv import vae_conv
from data.data_loader import load_data
from util.model_util import update_dropout_rate
from util.visual import plot_results


def get_model(parser):
    model_name = parser.model_name.lower()

    if model_name == 'vae_mlp':
        model = vae_mlp(parser.image_size * parser.image_size,
                        parser.intermediate_dim,
                        parser.latent_dim,
                        parser.dropout_rate)
    elif model_name == 'vae_conv':
        model = vae_conv(parser.image_size,
                         parser.filters,
                         parser.kernel_size,
                         parser.latent_dim,
                         parser.dropout_rate)
    else:
        raise NameError('Unknown model: {0}'.format(model_name))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='Load h5 model trained weights')
    parser.add_argument('-d', '--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('-s', '--image_size', type=int, default=28)
    parser.add_argument('-i', '--intermediate_dim', type=int, default=512)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-z', '--latent_dim', type=int, default=2)
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-f', '--filters', type=int, default=16)
    parser.add_argument('-k', '--kernel_size', type=int, default=3)
    parser.add_argument('-m', '--model_name', type=str, default='vae_conv')
    args = parser.parse_args()

    # get data
    if args.model_name == 'vae_conv':
        target_shape = [-1, args.image_size, args.image_size, 1]
    elif args.model_name == 'vae_mlp':
        target_shape = [-1, args.image_size * args.image_size]
    else:
        print('Unknown model name: ', args.model_name)
        exit(1)

    (x_train, y_train), (x_test, y_test) = load_data(target_shape)

    try:
        # get model
        (model, encoder, decoder) = get_model(args)
    except NameError as err:
        print(err)
        exit(1)

    plot_model(model,
               to_file='{0}.png'.format(model.name),
               show_shapes=True)

    if args.weights:
        model = model.load_weights(args.weights)
    else:
        # train the model
        model.fit(x_train,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  shuffle=True,
                  validation_data=(x_test, None))
        model.save_weights('{0}.h5'.format(model.name))

    _, _, z = encoder.predict(x_test, batch_size=args.batch_size)

    # update dropout rate == 0.0
    if not update_dropout_rate(decoder):
        print('Dropout layer is not exist')

    plot_results((encoder, decoder),
                 (x_test, y_test),
                 batch_size=args.batch_size,
                 model_name='{0}'.format(args.model_name))
