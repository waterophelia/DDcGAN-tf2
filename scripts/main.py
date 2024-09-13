import time
import os
import h5py
import numpy as np
from train import train
from scripts import generate

BATCH_SIZE = 24
EPOCHS = 1
LOGGING_PERIOD = 40
MODEL_SAVE_PATH = './model/'
IS_TRAINING = True

def main():
    if IS_TRAINING:
        print('\nBegin to train the network ...\n')
        with h5py.File('Training_Dataset.h5', 'r') as f:
            sources = f['data'][:]  # Load the training data
        train(sources, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, logging_period=LOGGING_PERIOD)
    else:
        print('\nBegin to generate pictures ...\n')
        test_path = './test_imgs/'
        save_path = './results/'
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Time = []
        for i in range(20):
            index = i + 1
            ir_path = os.path.join(test_path, f'IR{index}_ds.bmp')
            vis_path = os.path.join(test_path, f'VIS{index}.bmp')
            begin = time.time()
            model_path = os.path.join(MODEL_SAVE_PATH, 'model.ckpt')
            generate(ir_path, vis_path, model_path, index, output_path=save_path)
            end = time.time()
            Time.append(end - begin)
            print(f"pic_num:{index}")

        print(f"Time: mean: {np.mean(Time):.2f}s, std: {np.std(Time):.2f}s")

if __name__ == '__main__':
    main()
