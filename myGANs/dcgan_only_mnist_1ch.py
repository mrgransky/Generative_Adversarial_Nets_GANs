import argparse
import os
import sys
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataloader import Sentinel2Dataset

##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################

os.makedirs("images", exist_ok=True)

if os.path.expanduser('~') == "/home/farid":
	nc_files_path = os.path.join(os.path.expanduser('~'), 'datasets', 'sentinel2-l1c-random-rgb-image')
	rgb_dir = os.path.join(os.path.expanduser('~'), 'datasets', "sentinel2-l1c_RGB_IMGs")
else:
	dataset_dir = "/scratch/project_2004072" # scratch folder in my puhti account!
	tmp_dir = "/scratch/project_2004072/trashes"
	nc_files_path = os.path.join(dataset_dir, 'sentinel2-l1c-random-rgb-image')
	rgb_dir = os.path.join(dataset_dir, "sentinel2-l1c_RGB_IMGs")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batchSize", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# Device configuration
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.init_size = opt.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
		self.conv_blocks = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
			nn.Tanh(),
		)

	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True):
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block

		self.model = nn.Sequential(
			*discriminator_block(opt.channels, 16, bn=False),
			*discriminator_block(16, 32),
			*discriminator_block(32, 64),
			*discriminator_block(64, 128),
		)

		# The height and width of downsampled image
		ds_size = opt.img_size // 2 ** 4
		self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

	def forward(self, img):
		out = self.model(img)
		out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)
		return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# custom dataloader
dataset = Sentinel2Dataset(image_dir=rgb_dir)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=os.cpu_count())
print(len(dataloader), type(dataloader), dataloader)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
	for batch_idx, batch_images in enumerate(dataloader):
		print(epoch, batch_idx, batch_images.shape) # torch.Size([nb, Ch, 520, 520])

		# Adversarial ground truths
		valid = Tensor(batch_images.shape[0], 1).fill_(1.0).detach() # real images (1.0)
		fake = Tensor(batch_images.shape[0], 1).fill_(0.0).detach() # fake images (0.0)
		print(type(valid), valid.shape)
		print(type(fake), fake.shape)

		# Configure input
		real_batch_images = batch_images.type(Tensor)

		# Train Generator
		optimizer_G.zero_grad()
		z = Tensor(np.random.normal(0, 1, (batch_images.shape[0], opt.latent_dim)))
		gen_batch_images = generator(z)
		g_loss = adversarial_loss(discriminator(gen_batch_images), valid)
		g_loss.backward()
		optimizer_G.step()

		# Train Discriminator
		optimizer_D.zero_grad()
		real_loss = adversarial_loss(discriminator(real_batch_images), valid)
		fake_loss = adversarial_loss(discriminator(gen_batch_images.detach()), fake)
		d_loss = (real_loss + fake_loss) / 2
		d_loss.backward()
		optimizer_D.step()

		print(f"Epoch {epoch}/{opt.n_epochs} Batch {batch_idx}/{len(dataloader)} D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

		batches_done = epoch * len(dataloader) + batch_idx
		if batches_done % opt.sample_interval == 0:
			save_image(gen_batch_images.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)