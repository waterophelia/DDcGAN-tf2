import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

class Discriminator1(tf.keras.Model):
    def __init__(self, scope_name):
        super(Discriminator1, self).__init__(name=scope_name)  # Use the super class initializer
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation=None, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation=None, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation=None, name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(1, activation='tanh', use_bias=True, name='dense')

    def discrim(self, img):
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)

        out = self.conv1(img)
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.conv2(out)
        out = tf.keras.layers.BatchNormalization()(out, training=True)  # Apply BatchNorm
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.conv3(out)
        out = tf.keras.layers.BatchNormalization()(out, training=True)  # Apply BatchNorm
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.flatten(out)  # Automatically handle reshaping
        #print(f"Shape before dense layer: {out.shape}")  # Debugging statement to check shape

        out = self.dense_layer(out)
        out = out / 2 + 0.5  # Rescale output
        return out

class Discriminator2(tf.keras.Model):
    def __init__(self, scope_name):
        super(Discriminator2, self).__init__(name=scope_name)  # Use the super class initializer
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation=None, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation=None, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation=None, name='conv3')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()  # New global average pooling layer
        self.dense_layer = tf.keras.layers.Dense(1, activation='tanh', use_bias=True, name='dense')

    def discrim(self, img):
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)

        out = self.conv1(img)
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.conv2(out)
        out = tf.keras.layers.BatchNormalization()(out, training=True)  # Apply BatchNorm
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.conv3(out)
        out = tf.keras.layers.BatchNormalization()(out, training=True)  # Apply BatchNorm
        out = tf.nn.relu(out)  # Apply ReLU activation

        out = self.global_avg_pool(out)  # Use global average pooling instead of flattening
        #print(f"Shape before dense layer in Discriminator2: {out.shape}")  # Debugging statement to check shape

        out = self.dense_layer(out)
        out = out / 2 + 0.5  # Rescale output
        return out

def test_discriminators():
    # Create a dummy input tensor with batch size of 1, 64x64 image, and 1 channel
    dummy_input = tf.random.normal([1, 64, 64, 1])

    # Initialize Discriminator1
    d1 = Discriminator1(scope_name='disc1')
    output1 = d1.discrim(dummy_input)
    print("Discriminator1 Output:", output1.numpy())  # Use .numpy() to print the output in eager execution

    # Initialize Discriminator2
    d2 = Discriminator2(scope_name='disc2')
    output2 = d2.discrim(dummy_input)
    print("Discriminator2 Output:", output2.numpy())  # Use .numpy() to print the output in eager execution

# Run the test
test_discriminators()
