import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

from PIL import Image


def build_example(annotation, class_map, images_dir):
    img_path = os.path.join(images_dir, annotation['filename'].replace('set_01', '').replace(".xml", ".jpg"))
    # print("images_dir: " + images_dir)
    # print("annotation['filename']: " + annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    if img_raw[0] != 255 and img_raw[0] != 137:
        print(f"raw {img_raw[0]}")
        print(f"bad image {img_path}")
        raise Exception("bad image")

    try:
        width = int(annotation['size']['width'])
        height = int(annotation['size']['height'])
    except KeyError:
        im = Image.open(img_path)
        width, height = im.size

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
                print(f"weird name {obj['name']}")
                continue

            if float(obj['bndbox']['xmin']) > width or float(obj['bndbox']['xmin']) < 0:
                print(f"bad xmin {obj['bndbox']['xmin']}")
                continue

            if float(obj['bndbox']['ymin']) > height or float(obj['bndbox']['ymin']) < 0:
                print(f"bad ymin {obj['bndbox']['ymin']}")
                continue

            if float(obj['bndbox']['xmax']) > width or float(obj['bndbox']['xmax']) < 0:
                print(f"bad xmax {obj['bndbox']['xmax']}")
                continue

            if float(obj['bndbox']['ymax']) > height or float(obj['bndbox']['ymax']) < 0:
                print(f"bad ymax {obj['bndbox']['ymax']}")
                continue

            
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])

            try:
                truncated.append(int(obj['truncated']))
            except:
                pass

            try:
                views.append(obj['pose'].encode('utf8'))
            except:
                pass

    if len(classes) > 100:
        print(f"too many classes ({len(classes)}) on {img_path}")
        raise Exception(f"too many classes ({len(classes)}) on {img_path}")

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

        try:
            tf_example = build_example(annotation, class_map, images_dir)
            writer.write(tf_example.SerializeToString())
        except:
            pass

    writer.close()
    logging.info(f"Wrote {out_file}")

def pre_train(**kwargs):
    class_map = {name: idx for idx, name in enumerate(
        open(kwargs['classes']).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    parse_set(class_map, kwargs['output_train_file'], kwargs['train_labels_dir'], kwargs['train_data_dir'])
    parse_set(class_map, kwargs['output_val_file'], kwargs['val_labels_dir'], kwargs['val_data_dir'])

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