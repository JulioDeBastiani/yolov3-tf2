import os
import tensorflow as tf

from os import path
from absl import app, flags
from absl.flags import FLAGS
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2

from yolov3_tf2.dataset import preprocess_image, load_tfrecord_dataset


flags.DEFINE_string('dataset', None, 'path to the dataset, labels will be ignored')
flags.DEFINE_integer('size', 10, 'number of samples to take from the dataset')
flags.DEFINE_integer('input_size', 416, 'size of the input tensor (a square image with three channels')
flags.DEFINE_string('model_name', 'persondet', 'name of the model on tensorflow serving')
flags.DEFINE_string('input_tensor', 'input', 'name of the input tensor')

flags.mark_flag_as_required('dataset')


def main(argv):

    count = 0
    images = []
    files = [path.join(path.join(FLAGS.dataset, f))
        for f in os.listdir(FLAGS.dataset)
        if path.isfile(path.join(FLAGS.dataset, f))
    ]

    files = [f for f in files if f.endswith(('.png', 'jpg', 'jpeg'))]

    for file in files:
        img_raw = tf.image.decode_image(open(file, 'rb').read(), channels=3)
        image = preprocess_image(img_raw, FLAGS.input_size)
        image = tf.expand_dims(image, 0)
        images.append(image)

        count += 1

        if count == FLAGS.size:
            break

    input_tensor = tf.concat(images, 0)

    with tf.io.TFRecordWriter('tf_serving_warmup_requests') as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(
                name=FLAGS.model_name
            ),
            inputs={
                FLAGS.input_tensor: tf.make_tensor_proto(
                    input_tensor,
                    shape=input_tensor.shape,
                    dtype=input_tensor.dtype
                )
            }
        )

        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )

        writer.write(log.SerializeToString())
        print('"tf_serving_warmup_requests" created with success!')
        print('to use it paste it to the "<model>/<version>/assets.extra" folder on the serving configuration folder')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
