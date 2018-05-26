import argparse
from keras.utils import plot_model
from model.vae import vae
from data.data_loader import load_data
from util.visual import plot_results


def get_model(model_name, parser):
    model_name = model_name.lower()

    if model_name == 'vae':
        model = vae(parser.image_size * parser.image_size,
                    parser.intermediate_dim,
                    parser.latent_dim,
                    parser.mse)
    else:
        raise NameError('Unknown model: {0}'.format(model_name))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='Load h5 model trained weights')
    parser.add_argument('-m', '--mse', help='Use mse loss instead of binary cross entropy (default)',
                        action='store_true')
    parser.add_argument('-s', '--image_size', type=int, default=28)
    parser.add_argument('-i', '--intermediate_dim', type=int, default=512)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-z', '--latent_dim', type=int, default=2)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    args = parser.parse_args()

    # get data
    (x_train, y_train), (x_test, y_test) = load_data()

    try:
        # get model
        (model, encoder, decoder) = get_model('vae', args)
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
                  validation_data=(x_test, None),
                  verbose=2)
        model.save_weights('{0}.h5'.format(model.name))

    plot_results((encoder, decoder),
                 (x_test, y_test),
                 batch_size=args.batch_size,
                 model_name='{0}'.format(model.name))
