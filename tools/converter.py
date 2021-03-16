from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf

from yolov3_tf2.dataset import parse_set

flags.DEFINE_string('train_labels_dir', '/run/media/juju/backup_loja/set_03/train_label',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_string('val_labels_dir', '/run/media/juju/backup_loja/set_03/val_label',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_string('train_data_dir', '/run/media/juju/backup_loja/set_03/train_data',
                    'path to the train images')
flags.DEFINE_string('val_data_dir', '/run/media/juju/backup_loja/set_03/val_data',
                    'path to the validation images')
flags.DEFINE_string('output_train_file', './data/set_03_train.tfrecord', 'outpot dataset')
flags.DEFINE_string('output_val_file', './data/set_03_val.tfrecord', 'outpot dataset')
flags.DEFINE_string('classes', './data/person-det.names', 'classes file')

flags.DEFINE_bool('data_augmentation', False, 'Chooses if using data augmentation or not')


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    parse_set(
        class_map,
        FLAGS.output_train_file,
        FLAGS.train_labels_dir,
        FLAGS.train_data_dir,
        FLAGS.data_augmentation
    )
    parse_set(
        class_map,
        FLAGS.output_val_file,
        FLAGS.val_labels_dir,
        FLAGS.val_data_dir,
        FLAGS.data_augmentation
    )

    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
