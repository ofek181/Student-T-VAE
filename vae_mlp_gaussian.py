import tensorflow as tf
import tensorflow_probability as tfp


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


class VAE_MLP_G(tf.keras.Model):
    """
    Implements a fully connected architecture variational auto-encoder with Gaussian distribution.
    """
    def __init__(self, in_shape, enc_size, dec_size, l2):
        super(VAE_MLP_G, self).__init__()
        self.enc_size = enc_size
        self.dec_size = dec_size

        # creating the encoder
        input = tf.keras.Input(in_shape)
        x = input
        for layer in self.enc_size[:-1]:
            x = tf.keras.layers.Dense(layer,
                                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                                      activation='relu')(x)
        z_mu = tf.keras.layers.Dense(self.enc_size[-1], activation='linear')(x)
        z_var = tf.keras.layers.Dense(self.enc_size[-1], activation='softplus')(x)
        z = Reparameterization()([z_mu, z_var])

        self.encoder = tf.keras.models.Model(input, [z_mu, z_var, z], name='Encoder')

        # creating the decoder
        self.decoder = tf.keras.Sequential(name='Decoder')
        for layer in self.dec_size:
            self.decoder.add(tf.keras.layers.Dense(layer,
                                                   kernel_regularizer=tf.keras.regularizers.l2(l2),
                                                   activation=tf.nn.relu))

        self.decoder.add(tf.keras.layers.Dense(in_shape, name='output'))
        self.decoder.add(tfp.layers.IndependentBernoulli(in_shape, tfp.distributions.Bernoulli.logits))

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
        p_z = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.enc_size[-1]),
                                                                     scale=tf.ones(self.enc_size[-1])),
                                            reinterpreted_batch_ndims=1)
        return p_z

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: input data.
        :param training: boolean or boolean scalar tensor, indicating whether to
        run the `Network` in training mode or inference mode.
        :param mask: a mask or list of masks. A mask can be either a boolean tensor
              or None (no mask). For more details, check the guide
        :return: overrides the tensorflow call for the implemented VAE model.
        """
        z_mu, z_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)  # reconstructed images from the decoder
        # KL(q||p) = \int q(z)log(q(z)/p(z)) = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        q_z = self.posterior(z_mu, z_var)
        p_z = self.prior()
        self.kl = q_z.log_prob(z) - p_z.log_prob(z)  # General solution of the KL divergence.
        # self.kl = self.kl_loss(z_mu, z_var)
        return reconstructed

    @staticmethod
    def reconstruction_loss(x, x_v):
        """
        :param x: original data.
        :param x_v: reconstructed data.
        :return: reconstruction loss using binary cross entropy.
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        res = bce(x, x_v)
        return res

    def kl_loss(self):
        """
        General solution of the KL divergence.
        :return: KL(q||p) = \int q(z)log(q(z)/p(z))
        = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        """
        return tf.reduce_mean(self.kl)

    # @staticmethod
    # def kl_loss(z_mean, z_log_var):
    #     """
    #      Calculate the KL divergence for Gaussian distribution in a closed form.
    #     """
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #     return kl_loss

    def total_loss(self, x, x_hat):
        """
        :param x: original data.
        :param x_hat: reconstructed data.
        :return: objective of the VAE is to minimize the KL divergence + reconstruction loss.
        """
        reconstruction_loss = self.reconstruction_loss(x, x_hat)
        kl_loss = self.kl_loss()
        return reconstruction_loss + kl_loss

    def sample(self, sample_size):
        """
        :param sample_size: number of samples to generate.
        :return: sample from Gaussian distribution and decode synthetic data.
        """
        p_z = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.enc_size[-1]),
                                                                     scale=tf.ones(self.enc_size[-1])),
                                            reinterpreted_batch_ndims=1)
        z = p_z.sample(sample_size)
        reconstructed = self.decoder(z)
        return reconstructed.mean().numpy().reshape((sample_size, 28, 28))
