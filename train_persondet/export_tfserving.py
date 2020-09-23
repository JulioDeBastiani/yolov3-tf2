import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest


def unfreeze_models(**kwargs):
    if kwargs['tiny']:
        yolo = YoloV3Tiny(classes=kwargs['num_classes'])
    else:
        yolo = YoloV3(classes=kwargs['num_classes'])
    print()
    print()
    yolo.load_weights(kwargs['weights'])
    logging.info('weights loaded')

    tf.saved_model.save(yolo, kwargs['output'])
    logging.info("model saved to: {}".format(kwargs['output']))

    model = tf.saved_model.load(kwargs['output'])
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(os.path.join(os.getcwd(), kwargs['classes'])).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(os.path.join(os.getcwd(), kwargs['image']), 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    outputs = infer(img)
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs["yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))
