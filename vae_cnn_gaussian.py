import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Reparameterization(tf.keras.layers.Layer):
    """
    implements the re-parameterization trick for Gaussian distribution
    """

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: mean and variance of the Normal distribution.
        :param training: boolean or boolean scalar tensor, indicating whether to
        run the `Network` in training mode or inference mode.
        :param mask: a mask or list of masks. A mask can be either a boolean tensor
        or None (no mask). For more details, check the guide
        :return: overrides the tensorflow call for the implemented VAE model.
        """
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + z_var * epsilon


class VAE_CNN_G(tf.keras.Model):
    """
    Implements a convolutional architecture variational auto-encoder with Gaussian distribution.
    """
    def __init__(self, latent_dim):
        super(VAE_CNN_G, self).__init__()
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
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean", activation="linear")(x)
        z_var = tf.keras.layers.Dense(self.latent_dim, name="z_var", activation="softplus")(x)
        z = Reparameterization()([z_mean, z_var])
        self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_var, z], name="encoder")

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

    # @staticmethod
    # def kl_loss(z_mean, z_log_var):
    #     """
    #         Calculate the KL divergence for Gaussian distribution in a closed form.
    #     """
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #     return kl_loss

    def kl_loss(self):
        """
        General solution of the KL divergence.
        :return: KL(q||p) = \int q(z)log(q(z)/p(z))
        = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        """
        return tf.reduce_mean(self.kl)

    def total_loss(self, reconstruction_loss):
        """
        :param reconstruction_loss: reconstruction loss using binary cross entropy
        :return: reconstruction loss + kl loss
        """
        kl_loss = self.kl_loss()
        return reconstruction_loss + kl_loss

    @tf.function
    def train_step(self, inputs):
        """
        :param inputs: input data (image, image).
        :return: implements the train_step function of tensorflow to train the model.
        """
        data, _ = inputs
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            # KL(q||p) = \int q(z)log(q(z)/p(z)) = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
            q_z = self.posterior(z_mean, z_var)
            p_z = self.prior()
            self.kl = q_z.log_prob(z) - p_z.log_prob(z)  # General solution of the KL divergence.
            # kl_loss = self.kl_loss(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.reconstruction_loss(data, reconstruction)
            total_loss = self.total_loss(reconstruction_loss)

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
        z_mean, z_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = self.reconstruction_loss(data, reconstruction)
        # KL(q||p) = \int q(z)log(q(z)/p(z)) = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        q_z = self.posterior(z_mean, z_var)
        p_z = self.prior()
        z = q_z.sample()
        self.kl = q_z.log_prob(z) - p_z.log_prob(z)  # General solution of the KL divergence.
        # kl_loss = self.kl_loss(z_mean, z_log_var)
        val_loss = self.total_loss(reconstruction_loss)
        return {"loss": val_loss}

    @staticmethod
    def posterior(z_mu, z_var):
        """
        :param z_mu: mean of the Student's-T distribution.
        :param z_var: variance of the Student's-T distribution.
        :return: posterior of the Gaussian distribution.
        """
        q_z = tfp.distributions.Independent(tfp.distributions.Normal(loc=z_mu, scale=z_var),
                                            reinterpreted_batch_ndims=1)
        return q_z

    def prior(self):
        """
        :return: prior of the Gaussian distribution: N(0, I).
        """
        p_z = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.latent_dim),
                                                                     scale=tf.ones(self.latent_dim)),
                                            reinterpreted_batch_ndims=1)
        return p_z

    def sample(self):
        """
        :return: sampled images from the decoder.
        """
        results = np.zeros((28 * 8, 28 * 8))
        for i in range(8):
            for j in range(8):
                p_z = self.prior()
                z = p_z.sample(1)
                x_decoded = self.decoder.predict(z, verbose=0)
                digit = x_decoded[0].reshape(28, 28)
                results[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
        return results
