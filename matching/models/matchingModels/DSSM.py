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


class DSSM(Model):
    def __init__(self, user_num, item_num, embed_dim, user_mlp, item_mlp, activation='relu',
                 dnn_dropout=0., use_l2norm=False, loss_name="binary_cross_entropy_loss",
                 gamma=0.5, embed_reg=0., seed=None):
        """
        DSSM:经典双塔模型。
        Args:
            user_num：最大userindex + 1
            item_num：最大itemindex + 1
            embed_dim：用户和物品vector的embedding维度
            user_mlp:MLP形状
            item_mlp:MLP形状，最后一层大小和user tower必须一样
        """

        super(DSSM, self).__init__()
        if user_mlp[-1] != item_mlp[-1]:
            raise ValueError("The last value of user_mlp must be equal to item_mlp's.")
        # user embedding
        self.user_embedding_table = Embedding(input_dim=user_num,
                                              input_length=1,
                                              output_dim=embed_dim,
                                              embeddings_initializer='random_normal',
                                              embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding_table = Embedding(input_dim=item_num,
                                              input_length=1,
                                              output_dim=embed_dim,
                                              embeddings_initializer='random_normal',
                                              embeddings_regularizer=l2(embed_reg))
        # user_mlp_layer
        self.user_mlp_layer = MLP(user_mlp, activation, dnn_dropout)
        # item_mlp_layer
        self.item_mlp_layer = MLP(item_mlp, activation, dnn_dropout)
        self.use_l2norm = use_l2norm
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_info = self.user_embedding_table(tf.reshape(inputs['user'], [-1, ]))  # (None, embed_dim)
        # item info
        pos_info = self.item_embedding_table(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        neg_info = self.item_embedding_table(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # mlp
        user_info = self.user_mlp_layer(user_info)
        pos_info = self.item_mlp_layer(pos_info)
        neg_info = self.item_mlp_layer(neg_info)
        # norm
        if self.use_l2norm:
            user_info = tf.math.l2_normalize(user_info, axis=-1)
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
        # calculate similar scores.
        pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
        # add loss
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()