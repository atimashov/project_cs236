import torch
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET


class LandmarksDataset(Dataset):
    # The maximum person ID in  the dataset.
    MAX_LABEL = 1500
    IMAGE_SHAPE = 256, 128, 3
    def __init__(self, dir = '../datasets/dataset_100000', transform = None):
        self.dir = dir
        self.transform = transform

        self.filenames = [x for x in os.listdir(dir) if '.txt' in x]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Using readlines()
        with open('{}/{}'.format(self.dir, filename), 'r') as flabels:
            lines = flabels.readlines()
 
        count = 0
        # Strips the newline character
        lmrks = torch.zeros(len(lines) * 2)
        for i, line in enumerate(lines):
            x, y = line.strip().split(' ')
            lmrks[i], lmrks[i + len(lines)] = float(x) / 256 - 1, float(y)  / 256 - 1

        return lmrks


class LandmarksDataset_flow(Dataset):
    # The maximum person ID in  the dataset.
    MAX_LABEL = 1500
    IMAGE_SHAPE = 256, 128, 3
    def __init__(self, dir = '../datasets/dataset_100000', mode = 'train'):
        self.dir = dir
        self.mode = mode

        filenames = [x for x in os.listdir(dir) if '.txt' in x]
        L = len(filenames)
        if mode == 'train':
            self.filenames = filenames[: int(0.8 * L)]
        elif mode == 'val':
            self.filenames = filenames[int(0.8 * L):int(0.9 * L)]
        else:
            self.filenames = filenames[:L]



    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Using readlines()
        with open('{}/{}'.format(self.dir, filename), 'r') as flabels:
            lines = flabels.readlines()
 
        count = 0
        # Strips the newline character
        lmrks = torch.zeros(len(lines) * 2)
        for i, line in enumerate(lines):
            x, y = line.strip().split(' ')
            # lmrks[i], lmrks[i + len(lines)] = float(x) / 256 - 1, float(y)  / 256 - 1
            lmrks[2 * i], lmrks[2 * i + 1] = float(x) / 256 - 1, float(y)  / 256 - 1 # scale it to [-1, 1]

        return lmrks

class LandmarksDataset_68(Dataset):

    def __init__(self, transform=None):

        tree = ET.parse('../datasets/ibug_300W_large_face_landmark_dataset/labels_ibug_300W.xml')
        root = tree.getroot()

        # self.image_filenames = []
        self.landmarks = []
        self.crops = []
        # self.transform = transform
        # self.root_dir = 'ibug_300W_large_face_landmark_dataset'
        
        for filename in root[2]:
            # self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)
            crops = filename[0].attrib

            landmark = torch.zeros(2 * 68)
            for num in range(68):
                x = (int(filename[0][num].attrib['x']) - int(crops['left'])) / float(crops['width'])
                y = (int(filename[0][num].attrib['y']) - int(crops['top'])) / float(crops['height'])
                landmark[num], landmark[num + 68] = x, y
                # landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        # self.landmarks = np.array(self.landmarks).astype('float32')     

        # assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        
        # if self.transform:
        #     image, landmarks = self.transform(image, landmarks, self.crops[index])
        # print(self.crops[index])
        # landmarks = torch.tensor(landmarks) - 0.5

        return landmarks - 0.5


if __name__ == '__main__':
    # test 
    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

    
    # dataset = LandmarksDataset()
    dataset = LandmarksDataset_68()

    data_loader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers = 4)
    mins = []
    maxs = []
    i, j, k = 0, 0, 0
    for lmrks in data_loader:
        i += ((lmrks < 0).sum(dim = 1) > 0).sum()
        j += ((lmrks > 0).sum(dim = 1) > 0).sum()
        k += lmrks.shape[0]
        mins.append(lmrks.min())
        maxs.append(lmrks.max())
    print(min(mins), max(maxs))
    print(i, j, k)