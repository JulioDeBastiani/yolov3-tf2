import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

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
flags.DEFINE_string('classes', './data/set_00.names', 'classes file')


def build_example(annotation, class_map, images_dir):
    img_path = os.path.join(images_dir, annotation['filename'].replace('set_01', '').replace(".xml", ".jpg"))
    # print("images_dir: " + images_dir)
    # print("annotation['filename']: " + annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    width = width if width > 0 else 416
    height = height if height > 0 else 416

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            if not obj['name'] in class_map:
                continue
            
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def parse_set(class_map, out_file, annotations_dir, images_dir):
    writer = tf.io.TFRecordWriter(out_file)

    for annotation_file in tqdm.tqdm(os.listdir(annotations_dir)):
        if not annotation_file.endswith('.xml'):
            continue

        # print("file: " + annotation_file)
        annotation_xml = os.path.join(annotations_dir, annotation_file)
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map, images_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    logging.info(f"Wrote {out_file}")

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
