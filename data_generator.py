import tensorflow as tf


class DataGenerator:
    def __init__(
        self,
        mode,
        batch_size,
        resize_dim,
        shuffle=True,
        n_channels=3,
    ):
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.n_channels = n_channels
        self.load_dataset()

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.images, self.labels = (
            (x_train, y_train) if self.mode == "train" else (x_test, y_test)
        )

        self.index = tf.range((self.images).shape[0])

    def __len__(self):
        return int(tf.math.ceil((self.images).shape[0] / self.batch_size))

    def __call__(self):
        for i in self.index:
            x, y = self.preprocess(self.images[i]), self.labels[i]
            yield x, y

    def on_epoch_end(self):
        if self.shuffle == True:
            tf.random.shuffle(self.index)

    def preprocess(self, image):
        image = tf.image.resize(
            image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )
        image = image / 255.0
        image = tf.cast(image, tf.float32)
        return image
