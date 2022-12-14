# Implementation inspired by
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_tf_vit.py
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

import math

import tensorflow as tf
import tensorflow_addons as tfa


class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        projection = self.projection(inputs)
        shape = tf.shape(projection)
        embeddings = tf.reshape(tensor=projection, shape=(shape[0], shape[1] * shape[2], -1))
        return embeddings


class ViTEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, img_size, patch_size, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.img_size = img_size

        self.patch_embeddings = PatchEmbeddings(patch_size, hidden_size, name="patch_embeddings")
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.static_num_patches = (img_size // self.patch_size) * (img_size // self.patch_size)

        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_size),
            trainable=True,
            name="cls_token",
        )
        self.position_embeddings = self.add_weight(
            shape=(1, self.static_num_patches + 1, self.hidden_size),
            trainable=True,
            name="position_embeddings",
        )

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        embeddings_shape = tf.shape(embeddings)  # N, seq_len, dim

        current_num_patches = embeddings_shape[1] - 1  # without CLS

        if (
            current_num_patches == self.static_num_patches and height == width
        ):  # TODO YONIGO: constraint that original size is square
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed,
                shape=(
                    1,
                    int(math.sqrt(self.static_num_patches)),
                    int(math.sqrt(self.static_num_patches)),
                    embeddings_shape[2],
                ),
            ),
            size=(h0, w0),
            method="bicubic",
        )

        # shape = tf.shape(patch_pos_embed)
        # assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, embeddings_shape[2]))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  # N,H,W,C
        embeddings = self.patch_embeddings(inputs, training=training)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, inputs_shape[1], inputs_shape[2])
        embeddings = self.dropout(embeddings, training=training)

        return embeddings


class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ??\_(???)_/??
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(img_size, patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = tf.keras.Sequential(
            [
                Block(
                    num_heads,
                    attention_dim=hidden_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    mlp_dim=mlp_dim,
                    sd_survival_probability=(sd[i].numpy().item()),
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        x = self.blocks(x, training=training)
        x = self.norm(x)
        return x[:, 0]


def vit_tiny(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=192,
        depth=12,
        num_heads=3,
        mlp_dim=4 * 192,
        attention_bias=True,
        **kwargs,
    )
    return model


def vit_small(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_dim=4 * 384,
        attention_bias=True,
        **kwargs,
    )
    return model


def vit_base(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4 * 768,
        attention_bias=True,
        **kwargs,
    )
    return model
