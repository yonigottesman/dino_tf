import tensorflow as tf
import yaml
from tensorflow_addons.optimizers import AdamW

from data import dataset
from loss import DinoLoss
from model import build_dino


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate, steps, warmup_steps, alpha=0, name=None):
        super().__init__()
        self.base_learning_rate = tf.convert_to_tensor(base_learning_rate, dtype=tf.float32)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(base_learning_rate, steps - warmup_steps, alpha)
        self.warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay") as name:
            step = tf.cast(step, tf.float32)
            learning_rate = tf.cond(
                step < self.warmup_steps,
                lambda: tf.math.divide_no_nan(step, self.warmup_steps) * self.base_learning_rate,
                lambda: self.cosine_decay(step - self.warmup_steps),
                name=name,
            )
            return learning_rate


def train(config):
    ds = dataset(config)
    dino = build_dino(config, len(ds))
    loss = DinoLoss(
        out_dim=config["out_dim"],
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"],
    )

    optimizer = AdamW(
        weight_decay=WarmupCosineDecay(
            config["weight_decay"],
            config["epochs"] * len(ds),
            0,
            config["weight_decay_end"] / config["weight_decay"],
        ),
        learning_rate=WarmupCosineDecay(
            config["lr"]
            * (config["batch_size_per_gpu"] * tf.distribute.get_strategy().num_replicas_in_sync)
            / 256.0,  # linear scaling rule,
            config["epochs"] * len(ds),
            config["warmup_epochs"] * len(ds),
            config["min_lr"] / config["lr"],
        ),
        clipnorm=config["clip_grad"],
    )
    dino.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    dino.fit(x=ds, epochs=config["epochs"], steps_per_epoch=1)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
