import tensorflow as tf
import tensorflow_probability as tfp


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


class VAE_MLP_ST(tf.keras.Model):
    """
    Implements a fully connected architecture variational auto-encoder with Student's-T distribution.
    """
    def __init__(self, in_shape, enc_size, dec_size, l2):
        super(VAE_MLP_ST, self).__init__()
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

        z_nu = tf.keras.layers.Dense(self.enc_size[-1],
                                     kernel_regularizer=tf.keras.regularizers.l2(l2),
                                     kernel_initializer=tf.keras.initializers.RandomUniform(
                                         minval=2., maxval=30),
                                     activation='softplus',
                                     name='z_nu')(x)
        z = Reparameterization()([z_mu, z_var, z_nu])
        self.encoder = tf.keras.models.Model(input, [z_mu, z_var, z_nu, z], name='Encoder')

        # creating the decoder
        self.decoder = tf.keras.Sequential(name='Decoder')
        for layer in self.dec_size:
            self.decoder.add(tf.keras.layers.Dense(layer,
                                                   kernel_regularizer=tf.keras.regularizers.l2(l2),
                                                   activation=tf.nn.relu))

        self.decoder.add(tf.keras.layers.Dense(in_shape, name='output'))
        self.decoder.add(tfp.layers.IndependentBernoulli(in_shape, tfp.distributions.Bernoulli.logits))

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

    def prior(self, z_nu):
        """
        :param z_nu: degree of freedom.
        :return: prior of the Student's-T distribution: St(0, I, z_nu).
        """
        p_z = tfp.distributions.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.enc_size[-1]),
                                                                       scale=tf.ones(self.enc_size[-1]),
                                                                       df=z_nu), reinterpreted_batch_ndims=1)
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
        z_mu, z_var, z_nu, z = self.encoder(inputs)
        q_z = self.posterior(z_mu, z_var, z_nu)
        p_z = self.prior(z_nu)
        # KL(q||p) = \int q(z)log(q(z)/p(z)) = \int q(z)log(q(z)) - \int q(z)log(p(z)) = E(log(q(z))) - E(log(p(z)))
        self.kl = q_z.log_prob(z) - p_z.log_prob(z)  # General solution of the KL divergence.
        reconstructed = self.decoder(z)  # reconstructed images from the decoder
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
        :return: sample from Student's-T distribution and decode synthetic data.
        """
        p_z = tfp.distributions.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.enc_size[-1]),
                                                                       scale=tf.ones(self.enc_size[-1]),
                                                                       df=3*tf.ones(self.enc_size[-1])),
                                            reinterpreted_batch_ndims=1)
        z = p_z.sample(sample_size)
        reconstructed = self.decoder(z)
        return reconstructed.mean().numpy().reshape((sample_size, 28, 28))
