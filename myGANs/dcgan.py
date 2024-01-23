import argparse
import os
import sys
import random

import torch
import torchvision

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from dataloader import *
from utils import *
from networks import *
##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################

# how to run:
# in Puhti:
# python dcgan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc

# in Puota:
# python dcgan.py --rgbDIR $HOME/datasets/sentinel2-l1c_RGB_IMGs --resDIR $HOME/trash_logs/GANs/misc --batchSZ 64

# in Local laptop:
# python dcgan.py --rgbDIR /home/farid/datasets/sentinel2-l1c_RGB_IMGs --resDIR /home/farid/datasets/GANs_results/misc

# DCGAN ref link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#introduction

parser = argparse.ArgumentParser()
parser.add_argument('--numWorkers', type=int, default=16, help='number of cpu core(s)')
parser.add_argument('--batchSZ', type=int, default=4, help='input batch size')
parser.add_argument('--imgSZ', type=int, default=256, help='H & W input images') # can't change now!!
parser.add_argument('--imgNumCh', type=int, default=3, help='Image channel(s), def: 3 RGB')
parser.add_argument('--nz', type=int, default=100, help='noise latent z vector size')

parser.add_argument('--feature_g', type=int, default=256)
parser.add_argument('--feature_d', type=int, default=256)

parser.add_argument('--nepochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--spectralNormGen', type=bool, default=False, help='Spectrally Normalized Generator')
parser.add_argument('--spectralNormDisc', type=bool, default=True, help='Spectrally Normalized Discriminator')

parser.add_argument('--resDIR', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--rgbDIR', required=True, help='path to RGB dataset')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cudnn.benchmark: bool = True
nz = int(opt.nz) # dimension of the noise vector
nCh = int(opt.imgNumCh)
display_step: int = 250

opt.resDIR += f"_epoch_{opt.nepochs}"
opt.resDIR += f"_batch_SZ_{opt.batchSZ}"
opt.resDIR += f"_img_SZ_{opt.imgSZ}"
opt.resDIR += f"_latent_noise_SZ_{opt.nz}"
opt.resDIR += f"_lr_{opt.lr}"
opt.resDIR += f"_feature_g_{opt.feature_g}"
opt.resDIR += f"_feature_d_{opt.feature_d}"
opt.resDIR += f"_device_{device}"
opt.resDIR += f"_ngpu_{torch.cuda.device_count()}"
opt.resDIR += f"_display_step_{display_step}"
opt.resDIR += f"_numWorkers_{opt.numWorkers}"

if opt.spectralNormGen:
	opt.resDIR += f"_spectralNormGen_{opt.spectralNormGen}"

if opt.spectralNormDisc:
	opt.resDIR += f"_spectralNormDisc_{opt.spectralNormDisc}"

checkponts_dir = os.path.join(opt.resDIR, "checkpoints")
metrics_dir = os.path.join(opt.resDIR, "metrics")
models_dir = os.path.join(opt.resDIR, "models")
fake_imgs_dir = os.path.join(opt.resDIR, "fake_IMGs")
real_imgs_dir = os.path.join(opt.resDIR, "real_IMGs")

os.makedirs(opt.resDIR, exist_ok=True)
os.makedirs(checkponts_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fake_imgs_dir, exist_ok=True)
os.makedirs(real_imgs_dir, exist_ok=True)

# Specify the custom directory for PyTorch cache
os.environ['TORCH_HOME'] = models_dir

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
	num_workers=opt.numWorkers,
)
print(len(dataloader), dataloader)

print(f"Generator [spectral_norm: {opt.spectralNormGen}]".center(120, "-"))
netG = Generator(
	ngpu=torch.cuda.device_count(), 
	nz=int(opt.nz), 
	feature_g=int(opt.feature_g), 
	nCh=int(opt.imgNumCh),
	spectral_norm = opt.spectralNormGen,
).to(device)
netG.apply(weights_init)
print(netG)

print(f"Discriminator [spectral_norm: {opt.spectralNormDisc}]".center(120, "-"))
netD = Discriminator(
	ngpu=torch.cuda.device_count(), 
	feature_d=int(opt.feature_d), 
	nCh=int(opt.imgNumCh),
	spectral_norm = opt.spectralNormDisc,
).to(device)
netD.apply(weights_init)
print(netD)

print(
	f">> nParams:\t"
	f"Gen: {sum(p.numel() for p in netG.parameters() if p.requires_grad)} | "
	f"Disc: {sum(p.numel() for p in netD.parameters() if p.requires_grad)}"
)

print(f"inception_v3 [weights: DEFAULT]".center(120, "-"))
inception_model = torchvision.models.inception_v3(weights="DEFAULT", progress=False).to(device)
print(inception_model)

# loss fcn: since we have sigmoid at the final layer of Discriminator
criterion = torch.nn.BCELoss()

# optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

mean_generator_loss = 0
mean_discriminator_loss = 0
disc_losses = list()
gen_losses = list()

print(f"Training with {torch.cuda.device_count()} GPU(s) & {opt.numWorkers} CPU core(s)".center(100, " "))
for epoch in range(opt.nepochs):
	for batch_idx, batch_images in enumerate(dataloader):
		# print(epoch+1, batch_idx, type(batch_images), batch_images.shape)
		##################################
		# (1) Update Discriminator network 
		##################################
		
		# # train with real images
		# netD.zero_grad()
		# batch_images = batch_images.to(device)
		# cur_batch_size = batch_images.size(0)
		# disc_real_pred = netD(batch_images)
		# disc_loss_real = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
		
		# # train with fake generated images
		# fake_noise = torch.randn(cur_batch_size, nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
		# fake = netG(fake_noise)
		# disc_fake_pred = netD(fake.detach())
		# disc_loss_fake = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

		# disc_loss = 0.5 * (disc_loss_real + disc_loss_fake) # Discriminator loss of single batch

		# mean_discriminator_loss += disc_loss.item() / display_step

		# disc_loss.backward(retain_graph=True)
		# optimizerD.step()
		
		# ##############################
		# # (2) Update Generator network
		# ##############################
		# netG.zero_grad()
		# fake_noise_2 = torch.randn(cur_batch_size, nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
		# fake_2 = netG(fake_noise_2)
		# disc_fake_pred = netD(fake_2)
		# gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
		# gen_loss.backward()
		# optimizerG.step()
		
		# mean_generator_loss += gen_loss.item() / display_step

		# disc_losses.append(disc_loss.item())
		# gen_losses.append(gen_loss.item())

		if ((batch_idx+1) % display_step == 0) or (batch_idx+1 == len(dataloader)):
			print(
				f"Epoch {epoch+1}/{opt.nepochs} Batch {batch_idx+1}/{len(dataloader)} "
				# f"D_loss[batch]: {disc_loss.item():.6f} G_loss[batch]: {gen_loss.item():.6f} "
				# f"D_loss[avg]: {mean_discriminator_loss:.6f} G_loss[avg]: {mean_generator_loss:.6f}"
			)
			# vutils.save_image(batch_images, os.path.join(real_imgs_dir, f"real_samples_ep_{epoch+1}_batchIDX_{batch_idx+1}.png") , normalize=True)
			# vutils.save_image(fake.detach(), os.path.join(fake_imgs_dir, f"fake_samples_ep_{epoch+1}_batchIDX_{batch_idx+1}.png"), normalize=True)

			# mean_generator_loss = 0
			# mean_discriminator_loss = 0
	
# 	torch.save(netG.state_dict(), os.path.join(checkponts_dir, f"generator_ep_{epoch}.pth"))
# 	torch.save(netD.state_dict(), os.path.join(checkponts_dir, f"discriminator_ep_{epoch}.pth"))

# save_pickle(
# 	pkl=disc_losses, 
# 	fname=os.path.join(models_dir, f"{len(disc_losses)}_disc_losses.gz"),
# )

# save_pickle(
# 	pkl=gen_losses, 
# 	fname=os.path.join(models_dir, f"{len(gen_losses)}_gen_losses.gz"),
# )

plot_losses(disc_losses=disc_losses, gen_losses=gen_losses, saveDIR=metrics_dir)