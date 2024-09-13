import tensorflow as tf
from datetime import datetime
import time
import os
from models.generator import Generator
from models.discriminator import Discriminator1, Discriminator2
from utils.loss_functions import SSIM_LOSS, L1_LOSS, Fro_LOSS
from utils.data_preparation import prepare_training_data

# Training hyperparameters
PATCH_SIZE = 84
LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
EPS = 1e-8
RC = 4
BATCH_SIZE = 24
EPOCHS = 1  

def grad(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g

# Record start time
start_time = time.time()

def train(data_path, save_path, epochs=EPOCHS, batch_size=BATCH_SIZE, logging_period=1):
    start_time = datetime.now()
    print(f'Starting training with {epochs} epochs and batch size {batch_size}.')

    # Prepare data
    dataset, num_imgs = prepare_training_data(data_path, batch_size, PATCH_SIZE, RC)
    n_batches = num_imgs // batch_size
    print(f'Number of images: {num_imgs}, Batches per epoch: {n_batches}')

    # Initialize models
    generator = Generator('Generator')
    discriminator1 = Discriminator1('Discriminator1')
    discriminator2 = Discriminator2('Discriminator2')

    # Optimizers
    optimizer_G = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    optimizer_D1 = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    optimizer_D2 = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    # Create checkpoint to save model
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator1=discriminator1, discriminator2=discriminator2, optimizer_G=optimizer_G, optimizer_D1=optimizer_D1, optimizer_D2=optimizer_D2)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=5)

    # Training loop
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        for batch_idx, (vis_batch, ir_batch) in enumerate(dataset):
            # Training step
            with tf.GradientTape(persistent=True) as tape:
                generated_img = generator.transform(vis=vis_batch, ir=ir_batch)
                g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

                # Discriminator outputs
                grad_of_vis = grad(vis_batch)
                d1_real = discriminator1.discrim(vis_batch)  
                d1_fake = discriminator1.discrim(generated_img)  
                d2_real = discriminator2.discrim(ir_batch)  
                d2_fake = discriminator2.discrim(generated_img_ds)  

                # Compute losses
                g_loss_gan_d1 = -tf.reduce_mean(tf.math.log(d1_fake + EPSILON))
                g_loss_gan_d2 = -tf.reduce_mean(tf.math.log(d2_fake + EPSILON))
                g_loss_gan = g_loss_gan_d1 + g_loss_gan_d2

                loss_ir = Fro_LOSS(generated_img_ds - ir_batch)
                loss_vis = L1_LOSS(grad(generated_img) - grad_of_vis)
                g_loss_norm = loss_ir + 1.2 * loss_vis
                g_loss = g_loss_gan + 0.8 * g_loss_norm

                d1_loss_real = -tf.reduce_mean(tf.math.log(d1_real + EPSILON))
                d1_loss_fake = -tf.reduce_mean(tf.math.log(1.0 - d1_fake + EPSILON))
                d1_loss = d1_loss_real + d1_loss_fake

                d2_loss_real = -tf.reduce_mean(tf.math.log(d2_real + EPSILON))
                d2_loss_fake = -tf.reduce_mean(tf.math.log(1.0 - d2_fake + EPSILON))
                d2_loss = d2_loss_real + d2_loss_fake

            # Compute gradients
            gradients_G = tape.gradient(g_loss, generator.trainable_variables)
            gradients_D1 = tape.gradient(d1_loss, discriminator1.trainable_variables)
            gradients_D2 = tape.gradient(d2_loss, discriminator2.trainable_variables)

            # Apply gradients
            optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))
            optimizer_D1.apply_gradients(zip(gradients_D1, discriminator1.trainable_variables))
            optimizer_D2.apply_gradients(zip(gradients_D2, discriminator2.trainable_variables))

            if (batch_idx + 1) % logging_period == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{n_batches}, "
                      f"G_loss: {g_loss.numpy():.4f}, D1_loss: {d1_loss.numpy():.4f}, D2_loss: {d2_loss.numpy():.4f}")
                
        # Save model checkpoint after each epoch
        checkpoint_manager.save()
        print(f'Model checkpoint saved for epoch {epoch+1}.')

        # Record end time
        end_time = time.time()

        # Calculate training time
        training_time = end_time - start_time

        print(f"Training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    data_path = "/Users/an.balcer/Desktop/DDcGAN/data/Training_Dataset.h5"
    save_path = "/Users/an.balcer/Desktop/DDcGAN/checkpoints/"  

    # Make sure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Call the train function
    train(data_path, save_path)
