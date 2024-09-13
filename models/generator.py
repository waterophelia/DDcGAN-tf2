import tensorflow as tf
from .deconv_layers import deconv_vis, deconv_ir

WEIGHT_INIT_STDDEV = 0.05

class Generator(tf.keras.Model):
    def __init__(self, scope_name):
        super(Generator, self).__init__(name=scope_name)
        print(f"Initializing Generator with scope: {scope_name}")
        self.encoder = Encoder(scope_name + '_encoder')
        self.decoder = Decoder(scope_name + '_decoder')

    def transform(self, vis, ir):
        # Assume deconv_ir and deconv_vis are compatible with TensorFlow 2.x
        IR = deconv_ir(ir, strides=[1, 1, 1, 1], scope_name='deconv_ir')
        VIS = deconv_vis(vis, strides=[1, 1, 1, 1], scope_name='deconv_vis')
        
        IR_resized = tf.image.resize(IR, [tf.shape(VIS)[1], tf.shape(VIS)[2]])

        # Concatenate VIS and IR_resized along the channel dimension
        img = tf.concat([VIS, IR_resized], axis=-1)
        code = self.encoder(img)
        generated_img = self.decoder(code)
        return generated_img

class Encoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Encoder, self).__init__(name=scope_name)
        print(f"Initializing Encoder with scope: {scope_name}")
        self.conv_layers = [
            tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv1_1'),
            tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='dense_block_conv1'),
            tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='dense_block_conv2'),
            tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='dense_block_conv3'),
            tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='dense_block_conv4'),
        ]

    def call(self, image):
        out = image
        for layer in self.conv_layers:
            out = layer(out)
            out = tf.keras.layers.BatchNormalization()(out)  # Adding batch normalization
            #print(f"Shape after layer {layer.name}: {out.shape}")  # Debugging statement
        return out

class Decoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Decoder, self).__init__(name=scope_name)
        print(f"Initializing Decoder with scope: {scope_name}")
        self.conv_layers = [
            tf.keras.layers.Conv2D(240, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv2_1'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv2_2'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv2_3'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv2_4'),
            tf.keras.layers.Conv2D(1, 3, padding='same', activation=None, kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STDDEV), name='conv2_5')
        ]

    def call(self, image):
        out = image
        for layer in self.conv_layers[:-1]:
            out = layer(out)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.nn.relu(out)
        
        out = self.conv_layers[-1](out)
        out = tf.nn.tanh(out) / 2 + 0.5  # Scaled tanh output
        return out

def conv2d(x, kernel, bias, dense=False, use_relu=True, scope=None, bn=True):
    with tf.name_scope(scope):
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)
        if bn:
            out = tf.keras.layers.BatchNormalization()(out, training=True)
        if use_relu:
            out = tf.nn.relu(out)
        if dense:
            out = tf.concat([out, x], axis=-1)
        return out

if __name__ == "__main__":
    print("Testing Generator...")
    ir_dummy = tf.random.uniform((1, 21, 21, 1), dtype=tf.float32)
    vis_dummy = tf.random.uniform((1, 84, 84, 1), dtype=tf.float32)
    generator = Generator('GeneratorScope')
    generated_image = generator(vis_dummy, ir_dummy)
    print("Generated image shape:", generated_image.shape)
