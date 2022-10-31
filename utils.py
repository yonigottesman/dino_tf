import tensorflow as tf


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


class StepTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch = tf.Variable(0)
        self.step = tf.Variable(0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch.assign(epoch)

    def on_batch_begin(self, step, logs=None):
        self.step.assign(step)


class WrappedLoss(tf.keras.losses.Loss):
    def __init__(self, loss, step_tracker, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.step_tracker = step_tracker

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred, self.step_tracker.epoch)
