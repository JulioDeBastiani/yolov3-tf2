from absl import app, flags, logging
from absl.flags import FLAGS
import os

from yolov3_tf2.dataset import parse_set

flags.DEFINE_string('train_labels_dir', None,
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_string('val_labels_dir', None,
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_string('train_data_dir', None,
                    'path to the train images')
flags.DEFINE_string('val_data_dir', None,
                    'path to the validation images')
flags.DEFINE_string('output_train_file', './data/set_03_trains.tfrecord',
                    'output train tfrecord')
flags.DEFINE_string('output_val_file', './data/set_03_vals.tfrecord',
                    'output validation tfrecord')
flags.DEFINE_string('classes', './data/person-det.names',
                    'classes file')

flags.DEFINE_bool('use_data_augmentation', False,
                  'Whether or not to increase the train dataset with image augmentation')

flags.mark_flags_as_required([
    'train_labels_dir', 'val_labels_dir', 'train_data_dir', 'val_data_dir'
])


def main(_argv):

    del _argv

    # Turns off gpu as it's not required
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    parse_set(
        class_map,
        FLAGS.output_train_file,
        FLAGS.train_labels_dir,
        FLAGS.train_data_dir,
        FLAGS.use_data_augmentation
    )
    parse_set(
        class_map,
        FLAGS.output_val_file,
        FLAGS.val_labels_dir,
        FLAGS.val_data_dir,
        False
    )

    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
