import os
import cv2
import numpy as np
from protopost import ProtoPost

from utils import b64_to_img
from download_models import download_models

download_models()

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))
PORT = int(os.getenv("PORT", 80))
PADDING = float(os.getenv("PADDING", 0.1))

FACE_PROTO = "models/deploy.prototxt.txt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"

#load model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def get_faces(frame):
  #run face detector
  #TODO: config options for these params?
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
  face_net.setInput(blob)
  output = np.squeeze(face_net.forward())

  ih, iw = frame.shape[0:2]

  faces = []

  #grab confidence levels and box bounds
  for i in range(output.shape[0]):
    confidence = float(output[i, 2])

    if confidence > MIN_CONFIDENCE:
      x1, y1, x2, y2 = (output[i, 3:7] * np.array([iw, ih, iw, ih])).astype("int")#.tolist()

      #calculate padding
      pw, ph = (np.array([x2-x1, y2-y1]) * PADDING).astype("int")

      #pad and clip
      x1, x2 = np.clip([x1-pw, x2+pw], 0, iw).tolist()
      y1, y2 = np.clip([y1-ph, y2+ph], 0, ih).tolist()

      #convert to xywh
      x, y, w, h  = x1, y1, x2-x1, y2-y1
      print(x, y, w, h)

      faces.append({
        "bounds": [x, y, w, h],
        "confidence": confidence
      })

  return faces

def handler(data):
  img = b64_to_img(data)
  faces = get_faces(img)
  return faces

routes = {
  "": handler,
}

ProtoPost(routes).start(PORT)
