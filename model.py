import tensorflow as tf


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


class Dino(tf.keras.Model):
    def __init__(
        self, teacher: tf.keras.Model, student: tf.keras.Model, momentum_scheduler, freeze_last_layer, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student = student
        self.step_tracker = StepTracker()
        self.momentum_scheduler = momentum_scheduler
        self.freeze_last_layer = tf.convert_to_tensor(freeze_last_layer)

    # wrap dino loss with WrappedLoss that tracks the epoch number. tf.keras.losses.Loss expect y_true, y_pred
    # and dino loss also expects epoch number
    def compile(self, **kwargs):
        kwargs["loss"] = WrappedLoss(kwargs["loss"], self.step_tracker)
        super().compile(**kwargs)

    # add StepTracker to callbacks to track epoch number and batch for usage in the dino loss and train_step
    def fit(self, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(self.step_tracker)
        kwargs["callbacks"] = callbacks
        super().fit(**kwargs)

    def _forward(self, data, training):
        teacher_output = multi_crop_forward(self.teacher, multi_crop_batched=data["global_crops"], training=training)
        student_global = multi_crop_forward(self.student, multi_crop_batched=data["global_crops"], training=training)
        student_local = multi_crop_forward(self.student, multi_crop_batched=data["local_crops"], training=training)
        student_output = tf.concat([student_global, student_local], axis=1)
        loss = self.compiled_loss(teacher_output, student_output)
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._forward(data, training=True)

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        m = self.momentum_scheduler(self.step_tracker.step)

        for param_teacher, param_student in zip(self.teacher.weights, self.student.weights):
            param_teacher.assign(param_teacher * m + param_student * (1 - m))

        return {m.name: m.result() for m in self.metrics}


def multi_crop_forward(model, multi_crop_batched, training=False):
    """multi_crop_batched is shaped (batch,crops,w,h,c).
    The function will reshape to (batch*crops,w,h,c) and pass the large batch through the model.
    """
    shape = tf.shape(multi_crop_batched)
    # reshape multi_crop_batched so that first dimension is ordered batch_crop1,batch_crop2...
    huge_batch = tf.reshape(
        tf.transpose(multi_crop_batched, perm=(1, 0, 2, 3, 4)), (shape[0] * shape[1], shape[2], shape[3], shape[4])
    )

    huge_output = model(huge_batch, training=training)

    multi_crop_output = tf.transpose(tf.reshape(tf.transpose(huge_output), (-1, shape[1], shape[0])))
    return multi_crop_output


def dino_head(in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
    inputs = tf.keras.layers.Input(shape=(in_dim,))
    x = inputs
    for _ in range(nlayers - 1):
        x = tf.keras.layers.Dense(hidden_dim)(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dense(bottleneck_dim)(x)

    x = tf.math.l2_normalize(x, axis=-1)
    x = tf.keras.layers.Dense(out_dim, use_bias=False, name="last_layer")(x)
    # TODO YONIGO: did not use WeightNormalization and _no_grad_trunc_normal_
    head = tf.keras.Model(inputs=inputs, outputs=x)
    return head


def backbone(config):
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, pooling="avg")


def build_dino(config, steps_per_epoch):
    teacher_backbone = backbone(config)
    teacher_backbone.trainable = False

    teacher = tf.keras.Sequential(
        [teacher_backbone, dino_head(in_dim=teacher_backbone.output_shape[-1], out_dim=config["out_dim"])]
    )

    student_backbone = backbone(config)
    student = tf.keras.Sequential(
        [student_backbone, dino_head(in_dim=student_backbone.output_shape[-1], out_dim=config["out_dim"])]
    )
    momentum_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        config["momentum_teacher"], steps_per_epoch * config["epochs"], 1 / config["momentum_teacher"]
    )

    dino = Dino(teacher, student, momentum_scheduler, freeze_last_layer=config["freeze_last_layer"])
    return dino
