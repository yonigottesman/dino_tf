import tensorflow as tf


class DinoLoss:
    def __init__(
        self,
        out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        self.center = tf.Variable(
            tf.ones(out_dim),
        )
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.teacher_temp_schedule = tf.concat(
            [
                tf.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                tf.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            ],
            axis=0,
        )

    def __call__(self, teacher_output, student_output, epoch):
        """compute dino loss.
        teacher_output shape is (batch_size, global_crops, features)
        teacher_output shape is (batch_size, global_crops+local_crops, features)
        """

        student_out = student_output / self.student_temp
        student_out = tf.unstack(student_out, axis=1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = tf.math.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = tf.unstack(teacher_out, axis=1)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = tf.nn.softmax_cross_entropy_with_logits(q, student_out[v])
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        batch_center = tf.reduce_sum(teacher_output, axis=[0, 1])
        batch_center = tf.distribute.get_replica_context().all_reduce(tf.distribute.ReduceOp.SUM, batch_center)
        batch_center = batch_center / (
            tf.cast(
                tf.reduce_prod(tf.shape(teacher_output)[:2]) * tf.distribute.get_strategy().num_replicas_in_sync,
                tf.float32,
            )
        )

        # ema update
        self.center.assign(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))
