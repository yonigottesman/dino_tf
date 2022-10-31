import tensorflow as tf
import yaml
from tensorflow_addons.optimizers import AdamW

from data import dataset
from loss import DinoLoss
from model import build_dino
from utils import StepTracker, WarmupCosineDecay, WrappedLoss


def train(config):
    ds = dataset(config)
    step_tracker = StepTracker()
    dino = build_dino(config, len(ds), step_tracker)

    loss = DinoLoss(
        out_dim=config["out_dim"],
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"],
    )
    loss = WrappedLoss(loss, step_tracker)

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

    dino.fit(x=ds, epochs=config["epochs"], callbacks=step_tracker)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
