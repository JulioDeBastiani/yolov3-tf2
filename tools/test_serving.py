import numpy as np
import requests
import sys
import cv2

# Allow numpy to fully print out the array as a string, skipping the ellipsis
np.set_printoptions(threshold=sys.maxsize) 

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
