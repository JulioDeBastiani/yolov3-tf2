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


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    parse_set(class_map, FLAGS.output_train_file, FLAGS.train_labels_dir, FLAGS.train_data_dir)
    parse_set(class_map, FLAGS.output_val_file, FLAGS.val_labels_dir, FLAGS.val_data_dir)

    # writer = tf.io.TFRecordWriter(FLAGS.output_file)
    # image_list = open(os.path.join(
    #     FLAGS.data_dir, 'ImageSets', 'Main', 'aeroplane_%s.txt' % FLAGS.split)).read().splitlines()
    # logging.info("Image list loaded: %d", len(image_list))
    # for image in tqdm.tqdm(image_list):
    #     name, _ = image.split()
    #     annotation_xml = os.path.join(
    #         FLAGS.data_dir, 'Annotations', name + '.xml')
    #     annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
    #     annotation = parse_xml(annotation_xml)['annotation']
    #     tf_example = build_example(annotation, class_map)
    #     writer.write(tf_example.SerializeToString())
    # writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
