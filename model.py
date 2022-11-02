import tensorflow as tf

import vit


class Dino(tf.keras.Model):
    def __init__(
        self, teacher: tf.keras.Model, student: tf.keras.Model, momentum_scheduler, step_tracker, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student = student
        self.momentum_scheduler = momentum_scheduler
        step_tracker = step_tracker

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


class DinoHead(tf.keras.Model):
    def __init__(self, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        in_dim = input_shape[-1]
        inputs = tf.keras.layers.Input(shape=(in_dim,))
        x = inputs
        for _ in range(self.nlayers - 1):
            x = tf.keras.layers.Dense(self.hidden_dim)(x)
            if self.use_bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("gelu")(x)
        x = tf.keras.layers.Dense(self.bottleneck_dim)(x)

        x = tf.math.l2_normalize(x, axis=-1)
        x = tf.keras.layers.Dense(self.out_dim, use_bias=False, name="last_layer")(x)
        # TODO YONIGO: did not use WeightNormalization and _no_grad_trunc_normal_
        self.head = tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs, training=None, mask=None):
        return self.head(inputs, training)


class HeadedBackbone(tf.keras.Model):
    def __init__(self, backbone, head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=None):
        x = self.backbone(inputs, training=training)
        x = self.head(x, training=training)
        return x


def build_model(config):
    if config["arch"] == "resnet50":
        backbone = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, pooling="avg")
    else:
        backbone = vit.__dict__[config["arch"]](
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            sd_survival_probability=1 - config["drop_path_rate"],
        )
    head = DinoHead(out_dim=config["out_dim"])
    return HeadedBackbone(backbone, head)


def build_dino(config, steps_per_epoch, step_tracker):
    teacher = build_model(config)
    teacher.trainable = False

    student = build_model(config)

    momentum_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        config["momentum_teacher"], steps_per_epoch * config["epochs"], 1 / config["momentum_teacher"]
    )

    dino = Dino(teacher, student, momentum_scheduler, step_tracker)
    return dino
