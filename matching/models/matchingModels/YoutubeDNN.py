import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
from models.losses import get_loss
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., use_batch_norm=False):
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.use_batch_norm = use_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        if self.use_batch_norm:
            x = self.bt(x)
        x = self.dropout(x)
        return x


class YoutubeDNN(Model):
    def __init__(self, item_num, embed_dim, user_mlp, activation='relu',
                 dnn_dropout=0., use_l2norm=False, neg_num=4, batch_size=512,
                 embed_reg=0., seed=None):
        super(YoutubeDNN, self).__init__()
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
        # user_mlp_layer
        self.user_mlp_layer = MLP(user_mlp, activation, dnn_dropout)
        self.use_l2norm = use_l2norm
        self.embed_dim = embed_dim
        self.item_num = item_num
        self.neg_num = neg_num
        self.batch_size = batch_size
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs, training=False):
        seq_embed = tf.nn.embedding_lookup(self.item_embedding_table, inputs['click_seq'])
        # mask
        mask = tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32)  # (None, seq_len)
        seq_embed = tf.multiply(seq_embed, tf.expand_dims(mask, axis=-1))
        # user_info
        user_info = tf.reduce_mean(seq_embed, axis=1)  # (None, embed_dim)
        # mlp
        user_info = self.user_mlp_layer(user_info)
        if user_info.shape[-1] != self.embed_dim:
            raise ValueError("The last hidden unit must be equal to the embedding dimension.")
        # norm
        if self.use_l2norm:
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        if training:
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
            # calculate similar scores.
            pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
            logits = tf.concat([pos_scores, neg_scores], axis=-1)
            return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(100,), dtype=tf.int32),  # suppose sequence length=1
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()