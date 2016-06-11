import numpy as np
import cv2

data = np.load('video-compressed.npy')

while True:
  for frame in data:
    frame = frame.reshape([360, 480, 1])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()