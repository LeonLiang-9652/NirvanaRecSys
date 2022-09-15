import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

class CapsuleNetwork(Layer):
    def __init__(self, embed_dim, seq_len, bilinear_type=0, num_interest=4, stop_grad=True):
        super(CapsuleNetwork, self).__init__()
        self.bilinear_type = bilinear_type
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.embed_dim = embed_dim
        self.stop_grad = stop_grad

    def build(self, input_shape):
        if self.bilinear_type >= 2:
            self.w = self.add_weight(
                shape=[1, self.seq_len, self.num_interest * self.embed_dim, self.embed_dim],
                initializer='random_normal',
                name='weights'
            )

    def call(self, hist_emb, mask):
        if self.bilinear_type == 0:
            hist_emb_hat = tf.tile(hist_emb, [1, 1, self.num_interest])  # (None, seq_len, num_inter * embed_dim)
        elif self.bilinear_type == 1:
            hist_emb_hat = Dense(self.dim * self.num_interest, activation=None)(hist_emb)
        else:
            u = tf.expand_dims(hist_emb, axis=2)  # (None, seq_len, 1, embed_dim)
            hist_emb_hat = tf.reduce_sum(self.w * u, axis=3)  # (None, seq_len, num_inter * embed_dim)
        hist_emb_hat = tf.reshape(hist_emb_hat, [-1, self.seq_len, self.num_interest, self.embed_dim])
        hist_emb_hat = tf.transpose(hist_emb_hat, [0, 2, 1, 3])  # (None, num_inter, seq_len, embed_dim)
        hist_emb_hat = tf.reshape(hist_emb_hat, [-1, self.num_interest, self.seq_len, self.embed_dim])
        if self.stop_grad:
            hist_emb_iter = tf.stop_gradient(hist_emb_hat)
        else:
            hist_emb_iter = hist_emb_hat  # (None, num_inter, seq_len, embed_dim)

        if self.bilinear_type > 0:
            self.capsule_weight = self.add_weight(
                shape=[tf.shape(hist_emb_hat)[0], self.num_interest, self.seq_len],
                initializer=tf.zeros_initializer()
            )
        else:
            self.capsule_weight = tf.random.truncated_normal(
                shape=[tf.shape(hist_emb_hat)[0], self.num_interest, self.seq_len],
                stddev=1.0)
        tf.stop_gradient(self.capsule_weight)

        for i in range(3):
            att_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])  # (None, num_inter, seq_len)
            paddings = tf.zeros_like(att_mask)

            capsule_softmax_weight = tf.nn.softmax(self.capsule_weight, axis=1)  # (None, num_inter, seq_len)
            capsule_softmax_weight = tf.where(tf.equal(att_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)  # (None, num_inter, 1, seq_len)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, hist_emb_iter)  # (None, num_inter, 1, embed_dim)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, keepdims=True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(hist_emb_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                self.capsule_weight = self.capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, hist_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.embed_dim])
        return interest_capsule



class MIND(Model):
    def __init__(self, item_num, embed_dim, seq_len=100, num_interest=4, stop_grad=True, label_attention=True,
                 neg_num=4, batch_size=512, embed_reg=0., seed=None):
        super(MIND, self).__init__()
        with tf.name_scope("Embedding_layer"):
            # item embedding
            self.item_embedding_table = self.add_weight(name='item_embedding_table',
                                                        shape=(item_num, embed_dim),
                                                        initializer='random_normal',
                                                        regularizer=l2(embed_reg),
                                                        trainable=True)
            # embedding bias
            self.embedding_bias = self.add_weight(name='embedding_bias',
                                                  shape=(item_num,),
                                                  initializer=tf.zeros_initializer(),
                                                  trainable=False)
        self.capsule_network = CapsuleNetwork(embed_dim, seq_len, 0, num_interest, stop_grad)
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.label_attention = label_attention
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.neg_num = neg_num
        self.batch_size = batch_size
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs, training=False):
        user_hist_emb = tf.nn.embedding_lookup(self.item_embedding_table, inputs['click_seq'])
        mask = tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32)  # (None, seq_len)
        user_hist_emb = tf.multiply(user_hist_emb, tf.expand_dims(mask, axis=-1))  # (None, seq_len, embed_dim)
        # capsule network
        interest_capsule = self.capsule_network(user_hist_emb, mask)  # (None, num_inter, embed_dim)

        if training:
            if self.label_attention:
                item_embed = tf.nn.embedding_lookup(self.item_embedding_table, tf.reshape(inputs['pos_item'], [-1, ]))
                inter_att = tf.matmul(interest_capsule, tf.reshape(item_embed, [-1, self.embed_dim, 1]))  # (None, num_inter, 1)
                inter_att = tf.nn.softmax(tf.pow(tf.reshape(inter_att, [-1, self.num_interest]), 1))

                user_info = tf.matmul(tf.reshape(inter_att, [-1, 1, self.num_interest]), interest_capsule)  # (None, 1, embed_dim)
                user_info = tf.reshape(user_info, [-1, self.embed_dim])
            else:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)
            # train, sample softmax loss
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=self.item_embedding_table,
                biases=self.embedding_bias,
                labels=tf.reshape(inputs['pos_item'], shape=[-1, 1]),
                inputs=user_info,
                num_sampled=self.neg_num * self.batch_size,
                num_classes=self.item_num
            ))
            # add loss
            self.add_loss(loss)
            return loss
        else:
            # predict/eval
            pos_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['pos_item'])  # (None, embed_dim)
            neg_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['neg_item'])  # (None, neg_num, embed_dim)

            if self.label_attention:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)
            else:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)

            # calculate similar scores.
            pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info),
                                       axis=-1)  # (None, neg_num)
            logits = tf.concat([pos_scores, neg_scores], axis=-1)
            return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()