import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from natsort import natsorted
import glob
import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################

class Sentinel2Dataset(Dataset):
	def __init__(self, img_dir, img_sz: int=256):
		self.img_dir = img_dir
		self.transform = transforms.Compose([
			transforms.Resize((img_sz, img_sz)),
			############################################################################
			# suggested by Google Bard & chatGPT:
			# transforms.Normalize((0.4336, 0.4326, 0.4284), (0.1969, 0.1951, 0.1925)),
			# transforms.RandomBrightness(0.2),
			# transforms.CenterCrop(img_sz), # difference with random cropping!
			# transforms.RandomContrast(0.2),
			# transforms.RandomHorizontalFlip(0.5),
			# transforms.RandomVerticalFlip(0.5),
			############################################################################
			transforms.ToTensor(),
		])

	def __len__(self):
		return len(list(filter(lambda x: x.endswith('.png'), os.listdir(self.img_dir))))

	def __getitem__(self, idx):
		image_path = os.path.join(self.img_dir, f'{idx:06d}.png')
		image = Image.open(image_path).convert('RGB')
		image = self.transform(image)
		image_name = f'{idx:06d}.png'
		return image, image_name

def get_rgb_images(nc_files_path: str="path/to/nc_files", rgb_dir: str="path/to/rgb_images"):
	NC_FILES = natsorted( glob.glob( nc_files_path + "/" + "*.nc" ) ) # 10K nc files (sorted by name)
	num_nc_files = len(NC_FILES)
	
	for idx_nc, file_nc in enumerate(NC_FILES):
		# print(idx_nc, file_nc, os.path.splitext(os.path.basename(file_nc))[0])
		nc = netCDF4.Dataset(file_nc) # <class 'netCDF4._netCDF4.Dataset'> print (nc.variables.keys())

		r = nc.variables['B04'][0] # np.float64
		g = nc.variables['B03'][0] # np.float64
		b = nc.variables['B02'][0] # np.float64

		masked_array = np.stack([r, g, b], axis=-1) / num_nc_files
		# print(type(masked_array), masked_array.shape, np.min(masked_array), np.max(masked_array), masked_array.dtype)
		data = np.array(masked_array)
		####### must be [0. .. 1.0] #######
		data /= np.max(data) 
		data = np.clip(data, 0, None)
		####### must be [0. .. 1.0] #######

		plt.imsave(
			fname=os.path.join(rgb_dir, f"{os.path.splitext(os.path.basename(file_nc))[0]}.png"), 
			arr=data,
		)