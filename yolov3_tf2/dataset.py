import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import albumentations as A
import random
import os
import hashlib
import lxml.etree
import tqdm
import cv2
from typing import DefaultDict

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size, max_yolo_boxes):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, max_yolo_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size, max_yolo_boxes):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size, max_yolo_boxes))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


def build_example(file_name, images_dir, xml_data_dict, bytes_image, key):

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=xml_data_dict['height'])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=xml_data_dict['width'])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            file_name.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
           file_name.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xml_data_dict['xmin'])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xml_data_dict['xmax'])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=xml_data_dict['ymin'])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=xml_data_dict['ymax'])),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=xml_data_dict['classes_text'])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=xml_data_dict['classes'])),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=xml_data_dict['difficult']))
    }))


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


def parse_set(class_map, out_file, annotations_dir, images_dir, use_dataset_augmentation):

    writer = tf.io.TFRecordWriter(out_file)

    for annotation_file in tqdm.tqdm(os.listdir(annotations_dir)):
        if not annotation_file.endswith('.xml'):
            continue

        annotation_xml = os.path.join(annotations_dir, annotation_file)
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']

        height, width = get_image_dimensions(annotation, images_dir)
        pascal_voc_dict = parse_pascal_voc(annotation, class_map, height, width)

        pascal_voc_annotation_dict = parse_pascal_voc(annotation, class_map, height, width)

        if not pascal_voc_annotation_dict:
            continue            

        if len(pascal_voc_dict['classes']) > 100:
            logging.error(f"too many classes ({len(pascal_voc_dict['classes'])}) on {annotation_xml}")
            continue

        raw_image, key = open_image(annotation, images_dir)
        if not raw_image:
            continue

        tf_example = build_example(annotation['filename'], images_dir, pascal_voc_dict, raw_image, key)
        writer.write(tf_example.SerializeToString())

        if use_dataset_augmentation:
            tf_examples = augment_image(annotation, images_dir, pascal_voc_dict, class_map)
            for tf_example in tf_examples:
                writer.write(tf_example.SerializeToString())

    writer.close()
    logging.info(f'Wrote {out_file}')


def get_image_dimensions(annotation, images_dir):

    img_path = os.path.join(
        images_dir, annotation['filename'].replace('set_01', '').replace(".xml", ".jpg")
    )

    try:
        width = int(annotation['size']['width'])
        height = int(annotation['size']['height'])
    except KeyError:
        im = cv2.imread(img_path)
        height, width, _ = im.shape
    width = width if width > 0 else 416
    height = height if height > 0 else 416

    return height, width


def parse_pascal_voc(annotation, class_map, height, width) -> DefaultDict:

    pascal_voc_dict: DefaultDict = DefaultDict(list)

    if 'object' in annotation:
        for obj in annotation['object']:
            if not obj['name'] in class_map:
                logging.warning(f"weird name {obj['name']}")
                return

            if float(obj['bndbox']['xmin']) > width or float(obj['bndbox']['xmin']) < 0:
                logging.warning(f"bad xmin {obj['bndbox']['xmin']}")
                return

            if float(obj['bndbox']['ymin']) > height or float(obj['bndbox']['ymin']) < 0:
                logging.warning(f"bad ymin {obj['bndbox']['ymin']}")
                return

            if float(obj['bndbox']['xmax']) > width or float(obj['bndbox']['xmax']) < 0:
                logging.warning(f"bad xmax {obj['bndbox']['xmax']}")
                return

            if float(obj['bndbox']['ymax']) > height or float(obj['bndbox']['ymax']) < 0:
                logging.warning(f"bad ymax {obj['bndbox']['ymax']}")
                return

            difficult = bool(int(obj['difficult']))
            pascal_voc_dict['difficult'].append(int(difficult))
            pascal_voc_dict['xmin'].append(float(obj['bndbox']['xmin']) / width)
            pascal_voc_dict['ymin'].append(float(obj['bndbox']['ymin']) / height)
            pascal_voc_dict['xmax'].append(float(obj['bndbox']['xmax']) / width)
            pascal_voc_dict['ymax'].append(float(obj['bndbox']['ymax']) / height)
            pascal_voc_dict['classes_text'].append(obj['name'].encode('utf8'))
            pascal_voc_dict['classes'].append(class_map[obj['name']])
            pascal_voc_dict['height'].append(height)
            pascal_voc_dict['width'].append(width)

    return pascal_voc_dict


def open_image(annotation, images_dir):

    img_path = os.path.join(
        images_dir, annotation['filename'].replace('set_01', '').replace(".xml", ".jpg")
    )
    try:
        raw_image = open(img_path, 'rb').read()
    except:
        logging.warning(f'Could not open {annotation["filename"]} at {images_dir}')
        return

    key = hashlib.sha256(raw_image).hexdigest()

    if raw_image[0] != 255 and raw_image[0] != 137:
        logging.warning(f"raw {raw_image[0]}")
        logging.warning(f"bad image {img_path}")
        return
    
    return raw_image, key


def augment_image(annotation, images_dir, xml_data_dict, class_map):

    build_examples: list = []
    img_path = os.path.join(
        images_dir, annotation['filename'].replace('set_01', '').replace(".xml", ".jpg")
    )

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bounding_boxes = get_bounding_boxes(xml_data_dict)
    transformations, aug_names = get_default_augmentation_pipeline()

    for transform, aug_name in zip(transformations, aug_names):

        aug_image = transform(image=image, bboxes=bounding_boxes)

        a = cv2.imencode('.jpg', cv2.cvtColor(aug_image['image'], cv2.COLOR_RGB2BGR))[1].tostring()

        key = hashlib.sha256(a).hexdigest()
        file_name = annotation['filename'].replace('.jpg', f'--{aug_name}.jpg')

        image_dict = build_augmented_image_dict(aug_image, xml_data_dict, class_map)
        build_examples.append(build_example(file_name, images_dir, image_dict, a, key))

    return build_examples


def build_augmented_image_dict(aug_image, xml_data_dict, class_map):

    bbox_dict = get_bbox_dict(aug_image['bboxes'], class_map)

    augmented_image_dict: DefaultDict = DefaultDict(list)

    augmented_image_dict['height'].append(aug_image['image'].shape[0])
    augmented_image_dict['width'].append(aug_image['image'].shape[1])
    augmented_image_dict['truncated'] = xml_data_dict['truncated']
    augmented_image_dict['views'] = xml_data_dict['views']
    augmented_image_dict['difficult'] = xml_data_dict['difficult']
    augmented_image_dict['classes_text'] = xml_data_dict['classes_text']

    # Union both dicts
    augmented_image_dict = {**augmented_image_dict, **bbox_dict}

    return augmented_image_dict


def get_bbox_dict(bbox_list, class_map) -> dict:

    bbox_dict: DefaultDict = DefaultDict(list)

    for bbox in bbox_list:
        bbox_dict['xmin'].append(bbox[0])
        bbox_dict['ymin'].append(bbox[1])
        bbox_dict['xmax'].append(bbox[2])
        bbox_dict['ymax'].append(bbox[3])
        bbox_dict['classes'].append(bbox[4])
    
    return bbox_dict


def get_bounding_boxes(xml_data_dict: dict) -> list:
    
    bounding_boxes: list = []

    for i in range(len(xml_data_dict['xmin'])):
        bounding_boxes.append([
            xml_data_dict['xmin'][i],
            xml_data_dict['ymin'][i],
            xml_data_dict['xmax'][i],
            xml_data_dict['ymax'][i],
            xml_data_dict['classes'][i]
        ])

    return bounding_boxes


def get_default_augmentation_pipeline() -> list:

    # p is probability we set it allways to be 1 as for now we don't want randomness here
    # albumentations uses the random python lib to set it's seed.
    random.seed(4)
    transformations: list = []

    # albumentations format is pascal_voc, but divided by height and width

    # Blur augmentation
    transformations.append(A.Compose([
        A.Blur(blur_limit=(3,3), always_apply=True, p=1)
        ],
        bbox_params=A.BboxParams(format='albumentations')
    ))

    # HorizontalFlip augmentation
    transformations.append(A.Compose([
        A.HorizontalFlip(p=1)
        ],
        bbox_params=A.BboxParams(format='albumentations')
    ))

    # Contrast Limited Adaptive Histogram Equalization augmentation
    transformations.append(A.Compose([
        A.CLAHE(always_apply=True, p=1)
        ],
        bbox_params=A.BboxParams(format='albumentations')
    ))

    # Sepia augmentation
    transformations.append(A.Compose([
        A.ToSepia(always_apply=True, p=1)
        ],
        bbox_params=A.BboxParams(format='albumentations')
    ))
    # GaussNoise augmentation, very hard to see with eyes as it gives noise to some pixels
    transformations.append(A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), mean=1, always_apply=True, p=1)
        ],
        bbox_params=A.BboxParams(format='albumentations')
    ))
    
    aug_names: list = [
        'Blur', 'HorizontalFlip', 'Contrast', 'Sepia', 'GaussNoise'
    ]

    return transformations, aug_names
