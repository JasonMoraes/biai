import os
import cv2
from torch.utils.data import Dataset as BaseDataset
import numpy as np

class Dataset(BaseDataset):
    CLASSES = ['Animal', 'MaskingBackground']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + '.png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        #print(self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        #print(self.images_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_COLOR)

        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == [0, 0, 255]) for v in self.class_values]
       # print(masks)

        for row in mask:
            for cell in row:
             #   print(cell)
                if cell[0] < 30 and cell[1] < 30 and cell[2] > 200:
                    cell[2] = cell[1] = cell[0] = True
                else:
                    cell[2] = cell[1] = cell[0] = False

        mask = np.stack(mask).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
