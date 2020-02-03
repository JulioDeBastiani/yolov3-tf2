import numpy as np
# import tensorflow as tf
import requests
import sys
import cv2
np.set_printoptions(threshold=sys.maxsize) # Allow numpy to fully print out the array as a string, skipping the ellipsis

# img = tf.image.decode_image(open("./data/grazziotin.jpg", 'rb').read(), channels=3)
# img = tf.expand_dims(img, 0)
# img = tf.image.resize(img, (416, 416))

img = cv2.imread("./data/grazziotin.jpg")
img = cv2.resize(img, (416, 416))
img = img / 255
img = np.expand_dims(img, 0)
arr = np.array2string(img, separator=',', formatter={'float_kind':lambda x: "%.8f" % x})
url = 'http://localhost:8501/v1/models/yolov3:predict'
data = '{"inputs": ' + arr + '}'

with open("Output.txt", "w") as out:
    out.write(data)

print(data)
response = requests.post(url, data=data)
print(response.content.decode('utf'))




# import numpy as np
# import tensorflow as tf
# import requests
# import sys
# np.set_printoptions(threshold=sys.maxsize) # Allow numpy to fully print out the array as a string, skipping the ellipsis

# img = tf.image.decode_image(open("./data/grazziotin.jpg", 'rb').read(), channels=3)
# img = tf.image.resize(img, (416, 416))
# img = tf.expand_dims(img, 0)
# img = img / 255
# arr = np.array2string(img.numpy(), separator=',')
# url = 'http://localhost:8501/v1/models/yolov3:predict'
# data = '{"instances": ' + arr + '}'
# print(data)
# response = requests.post(url, data=data)
# print(response.content())