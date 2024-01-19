import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dataloader import *
from utils import *
##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################
# how to run:
# in Puhti:
# python dcgan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc

# in Puota:
# python dcgan.py --rgbDIR $HOME/datasets/sentinel2-l1c_RGB_IMGs --resDIR $HOME/trash_logs/GANs/misc

# in Local laptop:
# python dcgan.py --rgbDIR /home/farid/datasets/sentinel2-l1c_RGB_IMGs --resDIR /home/farid/datasets/GANs_results/misc

# DCGAN ref link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#introduction

parser = argparse.ArgumentParser()
parser.add_argument('--batchSZ', type=int, default=4, help='input batch size')
parser.add_argument('--imgSZ', type=int, default=256, help='H & W input images') # can't change now!!
parser.add_argument('--imgNumCh', type=int, default=3, help='Image channel(s), def: 3 RGB')
parser.add_argument('--nz', type=int, default=100, help='noise latent z vector size')

parser.add_argument('--feature_g', type=int, default=256)
parser.add_argument('--feature_d', type=int, default=256)

parser.add_argument('--nepochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--resDIR', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--rgbDIR', required=True, help='path to RGB dataset')

opt = parser.parse_args()
print(opt)



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ngpu = torch.cuda.device_count() # 1
nWorkers: int = os.cpu_count()
cudnn.benchmark: bool = True
nz = int(opt.nz) # dimension of the noise vector
feature_g = int(opt.feature_g)
feature_d = int(opt.feature_d)
nCh = int(opt.imgNumCh)
display_step: int = 500

opt.resDIR += f"_epoch_{opt.nepochs}"
opt.resDIR += f"_batch_SZ_{opt.batchSZ}"
opt.resDIR += f"_img_SZ_{opt.imgSZ}"
opt.resDIR += f"_latent_noise_SZ_{opt.nz}"
opt.resDIR += f"_lr_{opt.lr}"
opt.resDIR += f"_feature_g_{opt.feature_g}"
opt.resDIR += f"_feature_d_{opt.feature_d}"
opt.resDIR += f"_device_{device}"
opt.resDIR += f"_ngpu_{ngpu}"
opt.resDIR += f"_display_step_{display_step}"
opt.resDIR += f"_nWorkers_{nWorkers}"

os.makedirs(opt.resDIR, exist_ok=True)



if os.path.expanduser('~') == "/users/alijanif":
	dataset_dir = "/scratch/project_2004072" # scratch folder in my puhti account!
	nc_files_path = os.path.join(dataset_dir, 'sentinel2-l1c-random-rgb-image')
else:	
	nc_files_path = os.path.join(os.path.expanduser('~'), 'datasets', 'sentinel2-l1c-random-rgb-image')

if not os.path.exists(opt.rgbDIR) or len(natsorted( glob.glob( opt.rgbDIR + "/" + "*.png" ) )) < int(1e+4):
	os.makedirs(opt.rgbDIR)
	print(f">> Getting RGB Images from NC files [might take a while] ...")
	get_rgb_images(nc_files_path=nc_files_path, rgb_dir=opt.rgbDIR)
else:
	print(f'Already settled with {len(natsorted( glob.glob( opt.rgbDIR + "/" + "*.png" ) ))} RGB images!')

print(f'>> Generating a dataloader for {len(natsorted( glob.glob( opt.rgbDIR + "/" + "*.png" ) ))} RGB images...')
# custom dataloader
dataset = Sentinel2Dataset(img_dir=opt.rgbDIR, img_sz=opt.imgSZ)
dataloader = torch.utils.data.DataLoader(
	dataset=dataset, 
	batch_size=opt.batchSZ, 
	shuffle=True, 
	num_workers=nWorkers,
)
print(len(dataloader), type(dataloader), dataloader)

def weights_init(m): # zero-centered Normal distribution with std 0.02.
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
		torch.nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			
			nn.ConvTranspose2d(in_channels=nz, out_channels=feature_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=feature_g * 8),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(in_channels=feature_g * 8, out_channels=feature_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_g * 4),
			nn.ReLU(inplace=True),
			
			nn.ConvTranspose2d(in_channels=feature_g * 4, out_channels=feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_g * 2),
			nn.ReLU(inplace=True),
			
			nn.ConvTranspose2d(in_channels=feature_g * 2, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_g),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(in_channels=feature_g, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_g),
			nn.ReLU(inplace=True),
			
			nn.ConvTranspose2d(in_channels=feature_g, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_g),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(in_channels=feature_g, out_channels=nCh, kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh() # output normalized to [-1, 1]

		)
	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input) # [nb, ch, 256, 256]
		return output

class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(

			nn.Conv2d(in_channels=nCh, out_channels=feature_d, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(in_channels=feature_d, out_channels=feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_d * 2),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(in_channels=feature_d * 2, out_channels=feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_d * 4),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(in_channels=feature_d * 4, out_channels=feature_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=feature_d * 8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(in_channels=feature_d * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
			nn.Sigmoid() # final probability through a Sigmoid activation function

		)
	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
			# print(f"desc: forward: raw: {output.shape}")
			# print(f"desc: forward: raw.view(-1, 1): {output.view(-1, 1).shape}")
			# print(f"desc: forward: raw.view(-1, 1).squeeze(1): {output.view(-1, 1).squeeze(1).shape}")
		return output.view(-1, 1).squeeze(1) # Removes singleton dimension (dimension with size 1)

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(f"Generator".center(100, "-"))
print(netG)

netD = Discriminator(ngpu=ngpu).to(device)
netD.apply(weights_init)
print(f"Discriminator".center(100, "-"))
print(netD)

# loss fcn: since we have sigmoid at the final layer of Discriminator
criterion = nn.BCELoss()

# optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

mean_generator_loss = 0
mean_discriminator_loss = 0
cur_step = 0
disc_losses = list()
gen_losses = list()

print(f"Training with {ngpu} GPU(s) & {nWorkers} CPU core(s)".center(100, " "))
for epoch in range(opt.nepochs):
	for batch_idx, batch_images in enumerate(dataloader):
		# print(epoch, batch_idx, type(batch_images), batch_images.shape)
		##################################
		# (1) Update Discriminator network 
		##################################
		
		# train with real
		netD.zero_grad()
		batch_images = batch_images.to(device)
		cur_batch_size = batch_images.size(0)
		disc_real_pred = netD(batch_images)
		disc_loss_real = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
		
		# train with fake
		fake_noise = torch.randn(cur_batch_size, nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
		fake = netG(fake_noise)
		disc_fake_pred = netD(fake.detach())
		disc_loss_fake = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

		disc_loss = 0.5 * (disc_loss_real + disc_loss_fake) # Discriminator loss of single batch

		# Keep track of the average discriminator loss
		mean_discriminator_loss += disc_loss.item() / display_step

		disc_loss.backward(retain_graph=True)
		optimizerD.step()
		
		##############################
		# (2) Update Generator network
		##############################
		netG.zero_grad()
		fake_noise_2 = torch.randn(cur_batch_size, nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
		fake_2 = netG(fake_noise_2)
		disc_fake_pred = netD(fake_2)
		gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
		gen_loss.backward()
		optimizerG.step()
		
		# Keep track of the average generator loss
		mean_generator_loss += gen_loss.item() / display_step

		# Append losses to lists
		disc_losses.append(disc_loss.item())
		gen_losses.append(gen_loss.item())

		if batch_idx % display_step == 0 and batch_idx > 0:
			print(
				f"Epoch {epoch+1}/{opt.nepochs} Batch {batch_idx}/{len(dataloader)} "
				f"D_loss[batch]: {disc_loss.item():.6f} G_loss[batch]: {gen_loss.item():.6f} "
				f"D_loss[avg]: {mean_discriminator_loss:.6f} G_loss[avg]: {mean_generator_loss:.6f}"
			)
			vutils.save_image(batch_images, os.path.join(opt.resDIR, f"real_samples_ep_{epoch+1}_batchIDX_{batch_idx}.png") , normalize=True)
			vutils.save_image(fake.detach(), os.path.join(opt.resDIR, f"fake_samples_ep_{epoch+1}_batchIDX_{batch_idx}.png"), normalize=True)
			mean_generator_loss = 0
			mean_discriminator_loss = 0
	torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.resDIR, epoch))
	torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.resDIR, epoch))

plot_losses(disc_losses=disc_losses, gen_losses=gen_losses, saveDIR=opt.resDIR)