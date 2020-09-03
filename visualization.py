from glob import glob
from PIL import Image
import cv2
import numpy as np

def visualize_training_process():

    video_path = 'data/generated/vid.avi'
    video = cv2.VideoWriter(video_path, 0, 2, (640, 480))
    for i in range(0,10):
        path = f'data/generated/{i}/0.png'
        img = Image.open(path)
        img = img.convert('RGB')
        video.write(np.array(img))

    cv2.destroyAllWindows()
    video.release()

visualize_training_process()