import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        self.embed_dim = kwargs.pop("embed_dim")
        super(SelfAttention, self).__init__(**kwargs)
        self.query = Dense(self.embed_dim)
        self.key = Dense(self.embed_dim)
        self.value = Dense(self.embed_dim)
        self.out_proj = Dense(self.embed_dim)

    def call(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, V)
        return self.out_proj(output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class MultiHeadSelfAttention(Layer):
    def __init__(self, **kwargs):
        self.embed_dim = kwargs.pop("embed_dim")
        self.num_heads = kwargs.pop("num_heads")
        self.projection_dim = self.embed_dim // self.num_heads

        super(MultiHeadSelfAttention, self).__init__(**kwargs)

        self.query_dense = Dense(self.embed_dim)
        self.key_dense = Dense(self.embed_dim)
        self.value_dense = Dense(self.embed_dim)
        self.combine_heads = Dense(self.embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        Q = self.query_dense(inputs)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, V)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config
