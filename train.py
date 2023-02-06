import os
import matplotlib.pyplot as plt
from vae_mlp_st import *
from vae_mlp_gaussian import *
from vae_cnn_st import *
from vae_cnn_gaussian import *
from tensorflow.keras.callbacks import Callback
import argparse

filepath = os.path.dirname(os.path.abspath(__file__))


class callback(Callback):
    """
    Implements a tensorflow callback that saves samples every N epochs.
    """
    def __init__(self, dataset, distribution, architecture, vae, N):
        self.dataset = dataset
        self.distribution = distribution
        self.architecture = architecture
        self.vae = vae
        self.N = N

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.N == 0:
            if self.architecture == 'mlp':
                samples = self.vae.sample(sample_size=64)
                nindex, height, width = samples.shape
                ncols = int(np.sqrt(nindex))
                nrows = nindex // ncols
                assert nindex == nrows * ncols
                results = (samples.reshape(nrows, ncols, height, width).swapaxes(1, 2).reshape(height * nrows,
                                                                                               width * ncols))
                plt.imshow(results, cmap='gray')
                plt.axis('off')
                plt.savefig(filepath + "/samples/" + f'{self.distribution}_' + f'{self.architecture}_' +
                            f'{self.dataset}_' + f'epoch_{epoch + 1}.png')
            if self.architecture == 'cnn':
                samples = self.vae.sample()
                plt.imshow(samples, cmap='gray')
                plt.axis('off')
                plt.savefig(filepath + "/samples/" + f'{self.distribution}_' + f'{self.architecture}_' +
                            f'{self.dataset}_' + f'epoch_{epoch + 1}.png')


def main(args):
    if args.dataset == 'mnist':
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    elif args.dataset == 'fashion_mnist':
        (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError('Dataset not implemented')

    if args.architecture == 'cnn':
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        x_test = np.expand_dims(x_test, -1).astype("float32") / 255
        if args.distribution == 'studentT':
            vae = VAE_CNN_ST(args.latent_dim)
        elif args.distribution == 'gaussian':
            vae = VAE_CNN_G(args.latent_dim)
        else:
            raise ValueError('distribution not implemented')
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=vae.total_loss)
        history = vae.fit(x=x_train, y=x_train, batch_size=args.batch_size, epochs=args.epochs,
                          validation_data=(x_test, x_test),
                          callbacks=[callback(args.dataset, args.distribution, args.architecture, vae, 50)])

    elif args.architecture == 'mlp':
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).astype('float32') / 255.
        if args.distribution == 'studentT':
            vae = VAE_MLP_ST(in_shape=x_train.shape[1], enc_size=[128, 64, 32, args.latent_dim],
                             dec_size=[32, 64, 128], l2=1e-3)
        elif args.distribution == 'gaussian':
            vae = VAE_MLP_G(in_shape=x_train.shape[1], enc_size=[128, 64, 32, args.latent_dim],
                            dec_size=[32, 64, 128], l2=1e-3)
        else:
            raise ValueError('distribution not implemented')
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=vae.total_loss)
        history = vae.fit(x=x_train, y=x_train, batch_size=args.batch_size, epochs=args.epochs,
                          validation_data=(x_test, x_test), callbacks=[callback(args.dataset, args.distribution,
                                                                                args.architecture, vae, 50)],
                          shuffle=True, verbose=1)
    else:
        raise ValueError('Only cnn and mlp architectures are valid')

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Train and test loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(["train loss", "test loss"])
    plt.savefig(filepath + "/loss/" + f'{args.architecture}_' + f'{args.distribution}_'
                + f'{args.dataset}_' + f'loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--architecture',
                        help='architecture of the model: cnn or mlp',
                        type=str,
                        default='cnn')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=512)
    parser.add_argument('--epochs',
                        help='number of iteration over the data.',
                        type=int,
                        default=500)
    parser.add_argument('--latent_dim',
                        help='latent dimension for the bottleneck',
                        type=int,
                        default=3)
    parser.add_argument('--lr',
                        help='learning rate.',
                        type=float,
                        default=3e-4)
    parser.add_argument('--distribution',
                        help='gaussian or studentT distribution',
                        type=str,
                        default='studentT')

    args = parser.parse_args()
    main(args)
