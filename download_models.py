import urllib.request
from os.path import exists
import os

def download_models():
  os.makedirs("models", exist_ok=True)

  FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
  FACE_PROTO = "models/deploy.prototxt.txt"
  if not exists(FACE_PROTO):
    urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)

  FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
  FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
  if not exists(FACE_MODEL):
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)


if __name__ == "__main__":
  download_models()
