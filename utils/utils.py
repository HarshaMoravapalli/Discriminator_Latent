#from cv2 import imread, imshow, waitKey
import numpy as np
from PIL import Image, ImageOps
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import os
from tqdm import tqdm
import cv2
import numpy as np



class Custom_dataset(Dataset):
    def __init__(self,img_dir,size=None,transform=None):
        self.size = size
        self.images = []
        for file in os.listdir(img_dir):
            for image in os.listdir(os.path.join(img_dir,file)):
                image_path = os.path.join(img_dir,file,image)
                self.images.append(image_path)
        self.len = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #image = Image.open(self.images[index])
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = image.transpose([])
        #print("add",np.min(image))
        #print("add",np.max(image))
        image = cv2.resize(image, (512,512), interpolation = cv2.INTER_AREA).transpose([2,0,1])
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        
        '''cv2.imshow("test", image)
        
        waitKey(0) '''
                
        #img = resize_with_padding(img, (512, 512))
        """image.thumbnail((512, 512))
        # print(img.size)
        delta_width = 512 - image.size[0]
        delta_height = 512 - image.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        image = ImageOps.expand(image, padding)
        #image.show()
        #print('utils',image.size
        image = np.array(image)
        
        image = image.reshape((512,512, -1))
        image = image.transpose([2, 0, 1])
        import matplotlib.pyplot as plt
        plt.imshow(imgs[0].permute(1,2,0).cpu().numpy())
        plt.savefig('1.png')"""
        #print('i',image.shape)
        #imshow("test", image.transpose([1, 2, 0]))
        
        #waitKey(0) 
        #print("img",image.shape)
        
        return image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


















"""train_data = Custom_dataset(img_dir='Data/train',size = 0.25)
train_loader = DataLoader(train_data,batch_size=1,shuffle=False)
print(len(train_loader))



for epoch in range(0,2):
    with tqdm(range(len(train_loader))) as pbar:
        for i,img in  zip(pbar, train_loader):
            print(i,img)
            
            break"""


"""rescaleFactor = [0.25] 
image = io.imread(self.images[index])
print(image.shape)
image = transform.rescale(image, 0.25, anti_aliasing=True,multichannel=True)
print(image.shape)
data_orig = image.transpose(2,0,1)
print(data_orig.shape)
img = Image.open(source_path)"""

"""def normalize(img, maxval, reshape=False):
    #Scales images to be roughly [-1024 1024]
    
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
    
    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :] 
    
    return img"""