# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Шурмель К.А.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# как модели самокодировщика для задачи понижения размерности данных
# Вариант 7

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2


class ImageCompressor:
    
    HIDDEN_SIZE = 42
    MAX_ERROR = 1500.0
    LEARNING_RATE = 0.00085

    @classmethod
    def split_into_blocks(cls, height, width):
        img = cls.load_image()
        blocks = []
        for i in range(height // cls.BLOCK_HEIGHT):
            for j in range(width // cls.BLOCK_WIDTH):
                block = img[
                    cls.BLOCK_HEIGHT * i : cls.BLOCK_HEIGHT * (i + 1),
                    cls.BLOCK_WIDTH * j : cls.BLOCK_HEIGHT * (j + 1),
                    :3
                ]
                blocks.append(block)
        return np.array(blocks)

    @classmethod
    def blocks_to_image_array(cls, blocks, height, width):
        image_array = []
        blocks_in_line = width // cls.BLOCK_WIDTH
        for i in range(height // cls.BLOCK_HEIGHT):
            for y in range(cls.BLOCK_HEIGHT):
                line = [
                    [
                        blocks[i * blocks_in_line + j, (y * cls.BLOCK_WIDTH * 3) + (x * 3) + color]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(cls.BLOCK_WIDTH)
                ]
                image_array.append(line)
        return np.array(image_array)

    @classmethod
    def display_image(cls, img_array):
        scaled_image = 1.0 * (img_array + 1) / 2
        plt.axis('off')
        plt.imshow(scaled_image)
        plt.show()

    @classmethod
    def load_image(cls):
        image = cv2.imread("home.bmp", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (2.0 * image) - 1.0

    @classmethod
    def get_image_dimensions(cls):
        img = cls.load_image()
        return img.shape[0], img.shape[1]

    @classmethod
    def initialize_layers(cls):
        limit = np.sqrt(6 / (cls.INPUT_SIZE + cls.HIDDEN_SIZE))  # Xavier initialization
        layer1 = tf.random.uniform((cls.INPUT_SIZE, cls.HIDDEN_SIZE), -limit, limit)
        layer2 = tf.transpose(layer1)  # Symmetric initialization
        return layer1, layer2

    @classmethod
    def train_model(cls):
        error = cls.MAX_ERROR + 1
        previous_error = float('inf')
        epoch = 0

        # Adaptive learning rate setup
        learning_rate = cls.LEARNING_RATE
        DECAY_RATE = 0.9  # Factor to reduce learning rate
        MIN_LEARNING_RATE = 1e-6

        layer1, layer2 = cls.initialize_layers()
        blocks = cls.generate_blocks()

        while error > cls.MAX_ERROR:
            error = 0
            epoch += 1

            for block in blocks:
                # Forward pass
                hidden_layer = tf.matmul(block, layer1)
                output_layer = tf.matmul(hidden_layer, layer2)

                # Calculate difference
                diff = output_layer - block

                # Backpropagation
                layer1 -= learning_rate * tf.matmul(tf.matmul(tf.transpose(block), diff), tf.transpose(layer2))
                layer2 -= learning_rate * tf.matmul(tf.transpose(hidden_layer), diff)

            # Calculate epoch error
            error = sum(tf.reduce_sum((block @ layer1 @ layer2 - block) ** 2) for block in blocks)

            # Adjust learning rate adaptively
            if error < previous_error:
                # Reduce less aggressively or increase slightly
                learning_rate = max(learning_rate * 1.05, cls.LEARNING_RATE)
            else:
                # Slow convergence; reduce learning rate
                learning_rate = max(learning_rate * DECAY_RATE, MIN_LEARNING_RATE)

            previous_error = error

            print(f'Epoch {epoch} - Error: {error:.2f} - Learning Rate: {learning_rate:.6f}')

        compression_ratio = (32 * cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH * cls.total_blocks()) / (
            (cls.INPUT_SIZE + cls.total_blocks()) * 32 * cls.HIDDEN_SIZE + 2)
        print(f'Compression Ratio: {compression_ratio}')
        return layer1, layer2
    
    @classmethod
    def mod_of_vector(cls, vector):
        return np.sqrt(np.sum(vector ** 2, axis=0))

    @classmethod
    def generate_blocks(cls):
        height, width = cls.get_image_dimensions()
        return cls.split_into_blocks(height, width).reshape(cls.total_blocks(), 1, cls.INPUT_SIZE)

    @classmethod
    def total_blocks(cls):
        height, width = cls.get_image_dimensions()
        return (height * width) // (cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH)

    @classmethod
    def compress_image(cls, block_height, block_width):

        cls.BLOCK_HEIGHT, cls.BLOCK_WIDTH = block_height, block_width
        cls.INPUT_SIZE = cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH * 3
        
        height, width = cls.get_image_dimensions()
        layer1, layer2 = cls.train_model()

        original_image = cls.load_image()
        compressed_blocks = [block @ layer1 @ layer2 for block in cls.generate_blocks()]
        compressed_image = np.clip(np.array(compressed_blocks).reshape(cls.total_blocks(), cls.INPUT_SIZE), -1, 1)

        cls.display_image(original_image)
        cls.display_image(cls.blocks_to_image_array(compressed_image, height, width))
        
if __name__ == '__main__':
    ImageCompressor.compress_image(8, 8)


# Где инициализируется модель