import os
import argparse
import random
import math

import cv2
import numpy as np


def create_random_triangle(output_file, img_size=32):
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for vertex_i in range(3):
        x, y = random.randint(0, img_size-1), random.randint(0, img_size-1)
        img[x, y, :] = 255

        # add gaussian noise
        for _ in range(5):
            noise_x, noise_y = int(np.random.normal(x, 1)), int(np.random.normal(y, 1))
            noise_x = min(noise_x, img_size-1)
            noise_x = max(noise_x, 0)
            noise_y = min(noise_y, img_size-1)
            noise_y = max(noise_y, 0)
            img[noise_x, noise_y, :] = 255


    cv2.imwrite(output_file, img)


def create_equilateral_triangle(output_file, img_size=32):
    img = np.zeros((img_size, img_size, 3), np.uint8)

    is_inside_picture = False
    while not is_inside_picture:
        x1, y1 = random.randint(0, img_size-1), random.randint(0, img_size-1)
        x2, y2 = random.randint(0, img_size-1), random.randint(0, img_size-1)

        # translate
        x2_translated, y2_translated = x2 - x1, y2 - y1

        # rotate
        if x1 == x2:
            angle = 90
        else:
            angle = math.atan(y2_translated / x2_translated)
        x2_rotated = x2_translated * math.cos(-angle) - y2_translated * math.sin(-angle)

        # the third vertex
        x3_rotated = x2_rotated / 2
        y3_rotated = abs(x2_rotated) * math.sqrt(3) / 2

        # rotate back
        x3_translated = x3_rotated * math.cos(angle) - y3_rotated * math.sin(angle)
        y3_translated = x3_rotated * math.sin(angle) + y3_rotated * math.cos(angle)

        # translate back
        x3 = int(x3_translated + x1)
        y3 = int(y3_translated + y1)

        # check if all the third vertex is inside the picture
        if 0 <= x3 and x3 <= img_size-1 and 0 <= y3 and y3 <= img_size-1:
            is_inside_picture = True
            break

        # the third vertex (under the x axis)
        x3_rotated = x2_rotated / 2
        y3_rotated = - abs(x2_rotated) * math.sqrt(3) / 2

        # rotate back
        x3_translated = x3_rotated * math.cos(angle) - y3_rotated * math.sin(angle)
        y3_translated = x3_rotated * math.sin(angle) + y3_rotated * math.cos(angle)

        # translate back
        x3 = int(x3_translated + x1)
        y3 = int(y3_translated + y1)

        # check if all the third vertex is inside the picture
        if 0 <= x3 and x3 <= img_size-1 and 0 <= y3 and y3 <= img_size-1:
            is_inside_picture = True


    vertices = [[x1, y1], [x2, y2], [x3, y3]]
    for vertex in vertices:
        x, y = vertex[0], vertex[1]
        img[x, y, :] = 255

        # add gaussian noise
        for j in range(5):
            noise_x, noise_y = int(np.random.normal(x, 1)), int(np.random.normal(y, 1))
            noise_x = min(noise_x, img_size-1)
            noise_x = max(noise_x, 0)
            noise_y = min(noise_y, img_size-1)
            noise_y = max(noise_y, 0)
            img[noise_x, noise_y, :] = 255

    cv2.imwrite(output_file, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--data_num', default=60000)
    parser.add_argument('--img_size', default=32)
    args = parser.parse_args()

    data_nums = {
        'train': args.data_num * 8 // 10,
        'valid': args.data_num // 10,
        'test': args.data_num // 10
    }
    for phase in data_nums.keys():
        dir_name = os.path.join(args.data_dir, phase)
        os.makedirs(dir_name, exist_ok=True)
        for i in range(data_nums[phase]):
            create_random_triangle(
                os.path.join(dir_name, 'random_{}.png'.format(i)), args.img_size
            )
            create_equilateral_triangle(
                os.path.join(dir_name, 'equilateral_{}.png'.format(i)), args.img_size
            )



