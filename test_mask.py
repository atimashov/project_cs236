from datasets import *
import numpy as np
import cv2

dataset = LandmarksDataset()
data_loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers = 4)

for lmrks in data_loader:
    for i in range(lmrks.shape[0]):
        img = np.ones((100, 100, 3)) * 255
        for j in range(70):
            x, y = (lmrks[i, j] + 1) * 50, (lmrks[i, j + 70] + 1) * 50 
            img = cv2.circle(img, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=2)
            cv2.imwrite('images/q_{}.jpeg'.format(i), img) 
    break