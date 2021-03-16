import tensorflow as tf
import PIL
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import tqdm

from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset

flags.DEFINE_string('tfrecord_path', None, 'Path to the tfrecord to load and test')

flags.mark_flag_as_required('tfrecord_path')


def main(argv) -> None:

    del argv
    dataset = load_tfrecord_dataset(FLAGS.tfrecord_path,'./data/coco.names', 416)
    i = 0
    for instance in dataset:
    
        img = cv2.cvtColor(instance[0].numpy(), cv2.COLOR_RGB2BGR)

        for bbox in instance[1]:
            if max(bbox) == 0:
                continue

            img = cv2.rectangle(
                img,
                (bbox[0] * instance[0].shape[0], bbox[1] * instance[0].shape[1]),
                (bbox[2] * instance[0].shape[0], bbox[3] * instance[0].shape[1]),
                (255, 0, 0),
                2
            ) 

        cv2.imwrite(f'./data/take_{i}.jpg', img)
        i+=1

if __name__ == '__main__' :
    app.run(main)
