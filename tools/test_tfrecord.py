from absl import app, flags
from absl.flags import FLAGS
from tqdm import tqdm
import cv2
import os

from yolov3_tf2.dataset import load_tfrecord_dataset

flags.DEFINE_string('tfrecord_path', None,
                    'Path to the tfrecord to load and test')
flags.DEFINE_string('output', './out',
                    'The folder for the images output.')

flags.DEFINE_integer('max_yolo_boxes', 100,
                    'The max limit of the box possible on yolo output.')

flags.DEFINE_enum('mode', 'person',
                  ['face', 'person'], 'which neural network to train.')

flags.mark_flag_as_required('tfrecord_path')


def main(argv) -> None:

    del argv

    # Turns off gpu as it's not required
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = load_tfrecord_dataset(FLAGS.tfrecord_path,f'./data/{FLAGS.mode}-det.names', 416, FLAGS.max_yolo_boxes)
    os.makedirs(FLAGS.output, exist_ok=True)
    i = 0

    for instance in tqdm(dataset):

        img = cv2.cvtColor(instance[0].numpy(), cv2.COLOR_RGB2BGR)

        for bbox in instance[1]:
            if max(bbox) == 0:
                continue

            img = cv2.rectangle(
                img,
                (int(bbox[0] * instance[0].shape[1]), int(bbox[1] * instance[0].shape[0])),
                (int(bbox[2] * instance[0].shape[1]), int(bbox[3] * instance[0].shape[0])),
                (255, 0, 0),
                2
            )

        cv2.imwrite(f'{FLAGS.output}/take_{i}.jpg', img)
        i+=1

if __name__ == '__main__' :
    app.run(main)
