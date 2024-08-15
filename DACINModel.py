import tensorflow as tf
from tensorflow.keras import layers
import os

class DACIN(object):
    def __init__(self, network_architecture, learning_rate, batch_size, label_dim, model_path=None):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_dim = label_dim
        self.model_path = model_path

        self.gen_mvs = self.generator_mvs()
        self.dis_mvs = self.discriminator_mvs()
        self.enc_clu = self.encoder_cluster()
        self.dec_clu = self.decoder_cluster()
        self.se_clu = self.self_expression()

        self.var_se_list = []
        for var in self.enc_clu.trainable_variables:
            self.var_se_list.append(var)
        for var in self.dec_clu.trainable_variables:
            self.var_se_list.append(var)
        for var in self.se_clu.trainable_variables:
            self.var_se_list.append(var)

        self.optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate)

    # Generator for mvs
    def generator_mvs(self):
        # Functional model
        inputs = tf.keras.Input(self.network_architecture['n_input']*2+self.label_dim)
        G_h1 = layers.Dense(self.network_architecture['n_gen_1'], activation='relu')(inputs)
        G_h2 = layers.Dense(self.network_architecture['n_gen_2'], activation='relu')(G_h1)
        G_prob = layers.Dense(self.network_architecture['n_input'], activation='sigmoid')(G_h2)
        model = tf.keras.Model(inputs=inputs, outputs=G_prob)
        return model

    # Discriminator for mvs
    def discriminator_mvs(self):
        # Functional model
        inputs = tf.keras.Input(self.network_architecture['n_input']*2)
        D_h1 = layers.Dense(self.network_architecture['n_dis_1'], activation='relu')(inputs)
        D_h2 = layers.Dense(self.network_architecture['n_dis_2'], activation='relu')(D_h1)
        D_prob = layers.Dense(self.network_architecture['n_input'], activation='sigmoid')(D_h2)
        model = tf.keras.Model(inputs=inputs, outputs=D_prob)
        return model

    def encoder_cluster(self):
        inputs = tf.keras.Input(self.network_architecture['n_input'])
        E_h1 = layers.Dense(self.network_architecture['n_enc_1'], activation='relu')(inputs)
        E_h2 = layers.Dense(self.network_architecture['n_enc_2'], activation='relu')(E_h1)
        z = layers.Dense(self.network_architecture['n_z'], activation=None)(E_h2)
        model = tf.keras.Model(inputs=inputs, outputs=z, name="enc_clu")
        return model

    def self_expression(self):
        inputs = tf.keras.Input(self.batch_size)
        outputs = layers.Dense(self.batch_size, activation=None)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="se_clu")
        return model

    def decoder_cluster(self):
        inputs = tf.keras.Input(self.network_architecture['n_z'])
        D_h1 = layers.Dense(self.network_architecture['n_dec_1'], activation='relu')(inputs)
        D_h2 = layers.Dense(self.network_architecture['n_dec_2'], activation='relu')(D_h1)
        x_rec = layers.Dense(self.network_architecture['n_input'], activation='sigmoid')(D_h2)
        model = tf.keras.Model(inputs=inputs, outputs=x_rec, name="dec_clu")
        return model

    def whole_network(self, X, M, H, C, is_training=False):
        # generator
        self.g_inputs = tf.concat(values=[X, M, C], axis=1)  # concatenate by colume, size becomes n * (dim*2)
        self.g_X = self.gen_mvs(self.g_inputs, training=is_training)
        # Combine with observed data_optdigits
        self.Hat_X = X * M + self.g_X * (1 - M)
        # discriminator
        self.d_inputs = tf.concat(values=[self.Hat_X, H], axis=1)  # concatenate by colume, size becomes n * (dim*2)
        self.D_prob = self.dis_mvs(self.d_inputs, training=is_training)
        # encoder for clustering
        self.z = self.enc_clu(self.Hat_X)
        self.z_c = self.se_clu(tf.transpose(self.z))
        self.z_c = tf.transpose(self.z_c)
        self.dec_x = self.dec_clu(self.z_c)


    def loss_optimizer(self, X, M):
        # generator loss
        self.g_loss_adversarial = -tf.reduce_mean((1 - M) * tf.math.log(self.D_prob + 1e-8))
        self.g_loss_cmp = tf.reduce_sum((M * X - M * self.g_X) ** 2) / tf.reduce_sum(M)
        self.g_loss_mvs = tf.reduce_sum(((1-M) * X - (1-M) * self.g_X) ** 2) / tf.reduce_sum(1-M)
        #self.g_loss = self.g_loss_adversarial + 100*self.g_loss_cmp + 10*self.g_loss_mvs
        self.g_loss = self.g_loss_adversarial + 100 * self.g_loss_cmp
        # discriminator loss
        self.d_loss = -tf.reduce_mean(M * tf.math.log(self.D_prob + 1e-8) + (1 - M) * tf.math.log(1. - self.D_prob + 1e-8))
        # autoencoder loss
        self.ae_loss_cmp = tf.reduce_sum((M * self.Hat_X - M * self.dec_x) ** 2) / tf.reduce_sum(M)
        self.ae_loss_mvs = tf.reduce_sum(((1-M) * self.Hat_X - (1-M) * self.dec_x) ** 2) / tf.reduce_sum(1-M)
        self.se_loss = tf.reduce_mean((self.z - self.z_c)**2)
        self.se_coef = self.se_clu.trainable_variables[0]
        self.se_coef = tf.convert_to_tensor(self.se_coef)
        self.se_loss_coef = tf.reduce_mean(self.se_coef ** 2)
        self.cluster_loss = self.ae_loss_cmp + self.ae_loss_mvs + self.se_loss + 100*self.se_loss_coef

    def train_step(self, X, M, H, C, is_training):
        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as dis_tape, tf.GradientTape(persistent=True) as cluster_tape:
            self.whole_network(X, M, H, C, is_training)
            self.loss_optimizer(X, M)
            for i in range(5):
                self.gradients_of_dis = dis_tape.gradient(self.d_loss, self.dis_mvs.trainable_variables)
                self.optimizer.apply_gradients(zip(self.gradients_of_dis, self.dis_mvs.trainable_variables))
            self.gradients_of_gen = gen_tape.gradient(self.g_loss, self.gen_mvs.trainable_variables)
            self.optimizer.apply_gradients(zip(self.gradients_of_gen, self.gen_mvs.trainable_variables))
            self.gradients_of_clu = cluster_tape.gradient(self.cluster_loss, self.var_se_list)
            self.optimizer.apply_gradients(zip(self.gradients_of_clu, self.var_se_list))

    def save_model(self):
        self.gen_mvs.save(self.model_path + "/gen_mvs", save_format='tf')
        self.dis_mvs.save(self.model_path + "/dis_mvs", save_format='tf')
        self.enc_clu.save(self.model_path + "/enc_clu", save_format='tf')
        self.dec_clu.save(self.model_path + "/dec_clu", save_format='tf')
        self.se_clu.save(self.model_path + "/se_clu", save_format='tf')
        self.se_clu.save_weights(self.model_path + "/se_weights", save_format='tf')


def train_model(DACIN, miss_data_x, data_m, data_h, data_cluster, training_epochs):
    # show the model structure
    # DACINLabelOneHot.gen_mvs.summary()
    # DACINLabelOneHot.dis_mvs.summary()
    # DACINLabelOneHot.enc_clu.summary()
    # DACINLabelOneHot.dec_clu.summary()
    # DACINLabelOneHot.se_clu.summary()
    # logs
    logdir = os.path.join("./logs/")
    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    for epoch in range(training_epochs):
        DACIN.train_step(miss_data_x, data_m, data_h, data_cluster, is_training=True)
        display_step = 100
        save_step = 99
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "g_loss=", "{:.9f}".format(DACIN.g_loss),
                  "d_loss=", "{:.9f}".format(DACIN.d_loss),
                  "cluster_loss=", "{:.9f}".format(DACIN.cluster_loss),
                  "g_loss_adversarial=", "{:.9f}".format(DACIN.g_loss_adversarial),
                  "g_loss_cmp=", "{:.9f}".format(DACIN.g_loss_cmp),
                  "g_loss_mvs=", "{:.9f}".format(DACIN.g_loss_mvs),
                  "ae_loss_cmp=", "{:.9f}".format(DACIN.ae_loss_cmp),
                  "ae_loss_mvs=", "{:.9f}".format(DACIN.ae_loss_mvs),
                  "se_loss=", "{:.9f}".format(DACIN.se_loss),
                  "se_loss_coef=", "{:.9f}".format(DACIN.se_loss_coef))
        if epoch % save_step == 0:
            DACIN.save_model()


def get_se_coef(model_path):
    se_clu = tf.keras.models.load_model(model_path + "/se_clu", compile=True)
    se_coef = se_clu.trainable_variables[0]
    se_coef = se_coef.numpy()
    return se_coef


def imp_res_get(miss_data_x, data_m, data_cluster, model_path):
    gen_mvs = tf.keras.models.load_model(model_path + "/gen_mvs", compile=False)
    g_inputs = tf.concat(values=[miss_data_x, data_m, data_cluster], axis=1)
    g_X = gen_mvs(g_inputs, training=False)
    imp_x = miss_data_x * data_m + g_X * (1 - data_m)
    return imp_x, g_X


