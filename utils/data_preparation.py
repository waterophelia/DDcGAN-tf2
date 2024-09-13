import h5py
import tensorflow as tf

def load_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        source_imgs = f['data'][:]
    return source_imgs

def preprocess_images(source_imgs, patch_size, rc):
    vis_batch = tf.expand_dims(source_imgs[..., 0], axis=-1)
    ir_or_batch = tf.expand_dims(source_imgs[..., 1], axis=-1)
    ir_batch = tf.image.resize(ir_or_batch, [patch_size // rc, patch_size // rc])
    return vis_batch, ir_batch

def create_dataset(source_imgs, batch_size, patch_size, rc):
    dataset = tf.data.Dataset.from_tensor_slices(source_imgs)
    dataset = dataset.shuffle(buffer_size=len(source_imgs))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: preprocess_images(x, patch_size, rc))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def prepare_training_data(file_path, batch_size, patch_size, rc):
    source_imgs = load_dataset(file_path)
    dataset = create_dataset(source_imgs, batch_size, patch_size, rc)
    print("Training data prepared.")
    return dataset, source_imgs.shape[0]

if __name__ == "__main__":
    file_path = "../data/Training_Dataset.h5"  # Ensure this path is correct
    batch_size = 32
    patch_size = 128
    rc = 4

    dataset, num_images = prepare_training_data(file_path, batch_size, patch_size, rc)
    print(f"Number of images: {num_images}")
    print(f"Dataset: {dataset}")
