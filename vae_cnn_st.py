import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Reparameterization(tf.keras.layers.Layer):
    """
    implements the reparameterization trick for Student's-T distribution.
    """

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: mean, variance and v of the Student's-T distribution.
        :param training: boolean or boolean scalar tensor, indicating whether to
        run the `Network` in training mode or inference mode.
        :param mask: a mask or list of masks. A mask can be either a boolean tensor
        or None (no mask). For more details, check the guide
        :return: overrides the tensorflow call for the implemented VAE model.
        """
        z_mean, z_var, z_nu = inputs
        p_z = tfp.distributions.Independent(tfp.distributions.StudentT(loc=tf.zeros(tf.shape(z_mean)[1]),
                                                                       scale=tf.ones(tf.shape(z_mean)[1]),
                                                                       df=z_nu), reinterpreted_batch_ndims=1)
        T = p_z.sample()
        return z_mean + z_var * T


class VAE_CNN_ST(tf.keras.Model):
    """
    Implements a convolutional architecture variational auto-encoder with Student's-T distribution.
    """
    def __init__(self, latent_dim):
        super(VAE_CNN_ST, self).__init__()
        self.latent_dim = latent_dim

        # creating the encoder
        encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', name='conv_1')(encoder_inputs)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)
        x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)
        x = tf.keras.layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_4')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_4')(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean", activation='linear')(x)
        z_var = tf.keras.layers.Dense(self.latent_dim, name="z_var", activation='softplus')(x)
        z_nu = tf.keras.layers.Dense(self.latent_dim, name="z_nu", activation='softplus')(x)
        z = Reparameterization()([z_mean, z_var, z_nu])
        self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_var, z_nu, z], name="encoder")

        # creating the decoder
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(3136, name='dense_1')(latent_inputs)
        x = tf.keras.layers.Reshape((7, 7, 64), name='Reshape_Layer')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=1, padding='same', name='conv_transpose_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
        x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, 1, padding='same',
                                                          activation='sigmoid', name='conv_transpose_4')(x)
        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    @staticmethod
    def reconstruction_loss(x, x_v):
        """
        :param x: original data.
        :param x_v: reconstructed data.
        :return: reconstruction loss using binary cross entropy.
        """
        loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_v), axis=(1, 2)))
        return loss

    def kl_loss(self, z_mean, z_var, z_nu):
        """
        General solution of the KL divergence.
        :return: KL(q||p) = \int q(z)log(q(z)/p(z))
        = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        :param z_mean: mean of the Student's-T distribution.
        :param z_var: variance of the Student's-T distribution.
        :param z_nu: degree of freedom.
        :return: KL divergence using the general case solution.
        """
        q_z = self.posterior(z_mean, z_var, z_nu)
        p_z = self.prior(z_nu, dim=tf.shape(z_mean)[1])
        z = q_z.sample()
        kl_loss = tf.reduce_mean(q_z.log_prob(z) - p_z.log_prob(z))
        return kl_loss

    @staticmethod
    def total_loss(reconstruction_loss, kl_loss):
        """
        :param reconstruction_loss: reconstruction loss between original and decoded images.
        :param kl_loss: kl divergence.
        :return: objective of the VAE is to minimize the reconstruction loss + kl loss.
        """
        return reconstruction_loss + kl_loss

    @staticmethod
    def posterior(z_mu, z_var, z_nu):
        """
        :param z_mu: mean of the Student's-T distribution.
        :param z_var: variance of the Student's-T distribution.
        :param z_nu: degree of freedom.
        :return: posterior of the Student's-T distribution.
        """
        return tfp.distributions.Independent(tfp.distributions.StudentT(loc=z_mu, scale=z_var, df=z_nu),
                                             reinterpreted_batch_ndims=1)

    @staticmethod
    def prior(z_nu, dim):
        """
        :param z_nu: degree of freedom.
        :param dim: dimension of the vector.
        :return: prior of the Student's-T distribution: St(0, I, z_nu).
        """
        p_z = tfp.distributions.Independent(tfp.distributions.StudentT(loc=tf.zeros(dim),
                                                                       scale=tf.ones(dim),
                                                                       df=z_nu),
                                            reinterpreted_batch_ndims=1)
        return p_z

    @tf.function
    def train_step(self, inputs):
        """
        :param inputs: input data (image, image).
        :return: implements the train_step function of tensorflow to train the model.
        """
        data, _ = inputs
        with tf.GradientTape() as tape:
            z_mean, z_var, z_nu, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.reconstruction_loss(data, reconstruction)
            kl_loss = self.kl_loss(z_mean, z_var, z_nu)
            total_loss = self.total_loss(reconstruction_loss, kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

    @tf.function
    def test_step(self, inputs):
        """
        :param inputs: input data (image, image).
        :return: implements the test_step function of tensorflow to test the model.
        """
        data, _ = inputs
        z_mean, z_var, z_nu, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = self.reconstruction_loss(data, reconstruction)
        kl_loss = self.kl_loss(z_mean, z_var, z_nu)
        val_loss = self.total_loss(reconstruction_loss, kl_loss)
        return {"loss": val_loss}

    def sample(self):
        """
        :return: sampled images from the decoder.
        """
        results = np.zeros((28 * 8, 28 * 8))
        for i in range(8):
            for j in range(8):
                p_z = self.prior(4*tf.ones(self.latent_dim), self.latent_dim)
                z = p_z.sample(1)
                x_decoded = self.decoder.predict(z, verbose=0)
                digit = x_decoded[0].reshape(28, 28)
                results[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
        return results
