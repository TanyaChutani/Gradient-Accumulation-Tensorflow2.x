import tensorflow as tf
import argparse


model = Classification(5)
model.build((1,32,32,3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metric=tf.keras.metrics.SparseCategoricalAccuracy(),
)
model.fit(
    train_dataset,
    steps_per_epoch=train_gen.__len__(),
    epochs=2,
    callbacks=callbacks,
)

