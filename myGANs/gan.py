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
## dcgan:
# python gan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc --lr 0.0002 --ganMethodIdx 0 --numWorkers 8 --nepochs 50 --batchSZ 8 --dispInterval 100

## sngan:
# python gan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc --lr 0.0002 --ganMethodIdx 1 --numWorkers 8 --nepochs 50 --batchSZ 4 --dispInterval 100

# in Puota:
# python gan.py --rgbDIR $HOME/datasets/sentinel2-l1c_RGB_IMGs --resDIR $HOME/trash_logs/GANs/misc --batchSZ 64

# in Local laptop:
# python gan.py --rgbDIR /home/farid/datasets/sentinel2-l1c_RGB_IMGs --resDIR /home/farid/datasets/GANs_results/misc

# DCGAN ref link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#introduction

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=1, help='training epochs')
parser.add_argument('--batchSZ', type=int, default=4, help='input batch size')
parser.add_argument('--imgSZ', type=int, default=256, help='H & W input images') # can't change now!!
parser.add_argument('--imgNumCh', type=int, default=3, help='Image channel(s), def: 3 RGB')
parser.add_argument('--nz', type=int, default=100, help='noise latent z vector size')

parser.add_argument('--feature_g', type=int, default=256)
parser.add_argument('--feature_d', type=int, default=256)

parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--spectralNormGen', type=bool, default=False, help='Spectrally Normalized Generator')
parser.add_argument('--spectralNormDisc', type=bool, default=False, help='Spectrally Normalized Discriminator')

parser.add_argument('--resDIR', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--rgbDIR', required=True, help='path to RGB dataset')

parser.add_argument('--numWorkers', type=int, default=16, help='number of cpu core(s)')
parser.add_argument('--nGPUs', type=int, default=1, help='number of GPU(s)') # torch.cuda.device_count()
parser.add_argument('--cudaNum', type=int, default=0, help='CUDA') # torch.cuda.device_count()
parser.add_argument('--ganMethodIdx', type=int, default=0, help='GAN method')
parser.add_argument('--dispInterval', type=int, default=500, help='Display Interval')

opt = parser.parse_args()

device = torch.device(f"cuda:{opt.cudaNum}") if torch.cuda.is_available() else torch.device("cpu")
print(f">> Running Using device: {device}")

cudnn.benchmark: bool = True

GAN_METHODs: List[str] = ["dcgan", "sngan", "wgan"]

opt.resDIR += f"_{GAN_METHODs[opt.ganMethodIdx]}"
opt.resDIR += f"_epoch_{opt.nepochs}"
opt.resDIR += f"_batch_SZ_{opt.batchSZ}"
opt.resDIR += f"_img_SZ_{opt.imgSZ}"
opt.resDIR += f"_latent_noise_SZ_{opt.nz}"
opt.resDIR += f"_lr_{opt.lr}"
opt.resDIR += f"_feature_g_{opt.feature_g}"
opt.resDIR += f"_feature_d_{opt.feature_d}"
opt.resDIR += f"_device_{device}"
opt.resDIR += f"_ngpu_{opt.nGPUs}"
opt.resDIR += f"_display_step_{opt.dispInterval}"
opt.resDIR += f"_numWorkers_{opt.numWorkers}"

if GAN_METHODs[opt.ganMethodIdx] == "sngan":
	opt.spectralNormGen = True
	opt.spectralNormDisc = True

if opt.spectralNormGen:
	opt.resDIR += f"_spectralNormGen_{opt.spectralNormGen}"

if opt.spectralNormDisc:
	opt.resDIR += f"_spectralNormDisc_{opt.spectralNormDisc}"

print(opt)

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
os.makedirs(models_dir, exist_ok=True)

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

print(f'Dataloader: {len(natsorted( glob.glob( opt.rgbDIR + "/" + "*.png" ) ))} RGB images...')
dataset = Sentinel2Dataset(img_dir=opt.rgbDIR, img_sz=opt.imgSZ)
dataloader = torch.utils.data.DataLoader(
	dataset=dataset, 
	batch_size=opt.batchSZ,
	shuffle=True, 
	num_workers=opt.numWorkers,
	pin_memory=True,
)
print(f"dataset contans: {len(dataset)} images | dataloader with batch_size: {opt.batchSZ}: {len(dataloader)} batches")
#visualize(dataloader=dataloader)

def get_gen_disc_models(device: str="cuda"):
	print(f"Generator [spectral_norm: {opt.spectralNormGen}]".center(120, "-"))
	model_generator = Generator(
		ngpu=opt.nGPUs, 
		nz=int(opt.nz), 
		feature_g=int(opt.feature_g), 
		nCh=int(opt.imgNumCh),
		spectral_norm = opt.spectralNormGen,
	).to(device)
	get_param_(model=model_generator)
	
	print(f"Discriminator [spectral_norm: {opt.spectralNormDisc}]".center(120, "-"))
	model_discriminator = Discriminator(
		ngpu=opt.nGPUs, 
		feature_d=int(opt.feature_d), 
		nCh=int(opt.imgNumCh),
		spectral_norm = opt.spectralNormDisc,
	).to(device)
	get_param_(model=model_discriminator)
	return model_generator, model_discriminator

def test(dataloader, gen, disc, latent_noise_dim: int=100, device: str="cuda"):
	print(f"Test with {device}, {opt.nGPUs} GPU(s) & {opt.numWorkers} CPU core(s)".center(100, " "))
	gen.eval()
	disc.eval()

	print(f"inception_v3 [weights: DEFAULT]".center(120, "-"))
	inception_model = torchvision.models.inception_v3(weights="DEFAULT", progress=False).to(device)
	inception_model.eval()

	inception_model.fc = torch.nn.Identity() # remove fc or classification layer
	get_param_(model=inception_model)

	real_features_all, fake_features_all = get_real_fake_features(
		dataloader=dataloader, 
		model_generator=gen,
		model_inception_v3=inception_model,
		nz=latent_noise_dim,
		device=device,
	)
	mu_fake = torch.mean(fake_features_all, axis=0)
	mu_real = torch.mean(real_features_all, axis=0)
	sigma_fake = get_covariance(features=fake_features_all)
	sigma_real = get_covariance(features=real_features_all)
	with torch.no_grad(): # avoid storing intermediate gradient values,
		fid = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()
		print(f"FID: {fid:.3f}")

def train(init_gen_model=None, init_disc_model=None):
	print(f"Train with {opt.nGPUs} GPU(s) {device} | {opt.numWorkers} CPU core(s)".center(100, " "))

	if init_gen_model and init_disc_model:
		netG = init_gen_model		
		netD = init_disc_model
	else:
		netG, netD = get_gen_disc_models(device=device)

	netD.apply(weights_init)
	netG.apply(weights_init)
	
	# loss fcn: since we have sigmoid at the final layer of Discriminator
	criterion = torch.nn.BCELoss()

	# optimizer
	optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	disc_losses = list()
	gen_losses = list()
	mean_discriminator_losses = list()
	mean_generator_losses = list()

	best_discriminator_loss = float('inf')
	best_generator_loss = float('inf')

	best_discriminator_state_dict = None
	best_generator_state_dict = None

	for epoch in range(opt.nepochs):
		mean_discriminator_loss = 0
		mean_generator_loss = 0
		for batch_idx, (batch_images, batch_images_names) in enumerate(dataloader):
			# print(epoch+1, batch_idx, type(batch_images), batch_images.shape, batch_images_names)
			##################################
			# (1) Update Discriminator network 
			##################################

			# train with real images
			netD.zero_grad()
			batch_images = batch_images.to(device)
			cur_batch_size = batch_images.size(0)
			disc_real_pred = netD(batch_images)
			disc_loss_real = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
			
			# train with fake generated images
			# fake_noise = torch.randn(cur_batch_size, opt.nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images			
			fake_noise = torch.randn(cur_batch_size, opt.nz, device=device) # [nb, 100]
			fake = netG(fake_noise)
			disc_fake_pred = netD(fake.detach())
			disc_loss_fake = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

			disc_loss = 0.5 * (disc_loss_real + disc_loss_fake) # Discriminator loss of single batch

			mean_discriminator_loss += disc_loss.item()

			disc_loss.backward(retain_graph=True)
			optimizerD.step()
			
			##############################
			# (2) Update Generator network
			##############################
			netG.zero_grad()
			# fake_noise_2 = torch.randn(cur_batch_size, opt.nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
			fake_noise_2 = torch.randn(cur_batch_size, opt.nz, device=device) # [nb, 100]
			fake_2 = netG(fake_noise_2)
			disc_fake_pred = netD(fake_2)
			gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
			gen_loss.backward()
			optimizerG.step()
			
			mean_generator_loss += gen_loss.item()

			disc_losses.append(disc_loss.item())
			gen_losses.append(gen_loss.item())

			if ((batch_idx+1) % opt.dispInterval == 0) or (batch_idx+1 == len(dataloader)):
				print(
					f"Epoch {epoch+1}/{opt.nepochs} Batch {batch_idx+1}/{len(dataloader)} "
					f"D_loss[batch]: {disc_loss.item():.3f} G_loss[batch]: {gen_loss.item():.3f} "
				)
				real_batch_img_names = f"{'_'.join(batch_images_names[:-1])}_{batch_images_names[-1].split('.')[0]}"
				vutils.save_image(
					tensor=batch_images, 
					fp=os.path.join(real_imgs_dir, f"epoch{epoch+1}_batchIDX_{batch_idx+1}_Real_IMGs_{real_batch_img_names.replace('.png', '')}.png"), 
					normalize=True,
				)
				vutils.save_image(
					tensor=fake.detach(), 
					fp=os.path.join(fake_imgs_dir, f"fake_samples_ep_{epoch+1}_batchIDX_{batch_idx+1}.png"), 
					normalize=True,
				)

		# save models of all epochs
		torch.save(netG.state_dict(), os.path.join(checkponts_dir, f"Generator_model_epoch_{epoch+1}.pth"))
		torch.save(netD.state_dict(), os.path.join(checkponts_dir, f"Discriminator_model_epoch_{epoch+1}.pth"))

		# Calculate mean losses at the end of the epoch
		mean_discriminator_loss /= len(dataloader)
		mean_generator_loss /= len(dataloader)

		# Check if the current model has lower losses than the best
		if mean_discriminator_loss < best_discriminator_loss:
			print(
				f"Found better Discriminator model @ epoch: {epoch+1} "
				f"prev_best_loss: {best_discriminator_loss} "
				f"curr_loss: {mean_discriminator_loss:.3f}".center(150, " ")
			)
			best_discriminator_loss = mean_discriminator_loss
			best_disc_model = netD
			best_discriminator_state_dict = netD.state_dict()

		if mean_generator_loss < best_generator_loss:
			print(
				f"Found better Generator model @ epoch: {epoch+1} "
				f"prev_best_loss: {best_generator_loss} "
				f"curr_loss: {mean_generator_loss:.3f}".center(150, " ")
			)
			best_generator_loss = mean_generator_loss
			best_gen_model = netG
			best_generator_state_dict = netG.state_dict()
		
		print(f"\tEpoch {epoch+1}/{opt.nepochs} Mean D_loss: {mean_discriminator_loss:.3f} G_loss: {mean_generator_loss:.3f}")

		mean_discriminator_losses.append(mean_discriminator_loss)
		mean_generator_losses.append(mean_generator_loss)

	# Save the best models after training is complete
	torch.save(best_generator_state_dict, os.path.join(checkponts_dir, f"Generator_model_best.pth"))
	torch.save(best_discriminator_state_dict, os.path.join(checkponts_dir, f"Discriminator_model_best.pth"))

	save_pickle(
		pkl=disc_losses,
		fname=os.path.join(models_dir, f"disc_losses.gz"),
	)

	save_pickle(
		pkl=gen_losses,
		fname=os.path.join(models_dir, f"gen_losses.gz"),
	)

	save_pickle(
		pkl=mean_discriminator_losses,
		fname=os.path.join(models_dir, f"mean_disc_losses.gz"),
	)

	save_pickle(
		pkl=mean_generator_losses,
		fname=os.path.join(models_dir, f"mean_gen_losses.gz"),
	)

	return best_gen_model, best_disc_model

def main():
	init_gen_model, init_disc_model = get_gen_disc_models(device=device)		
	try:
		model_gen = init_gen_model
		model_disc = init_disc_model
		model_gen.load_state_dict(torch.load(os.path.join(checkponts_dir, f"Generator_model_best.pth")))
		model_disc.load_state_dict(torch.load(os.path.join(checkponts_dir, f"Discriminator_model_best.pth")))
		print("Loaded best generator and discriminator models successfully.")
	except Exception as e:
		print(f"<!> {e}")
		model_gen, model_disc = train(init_gen_model, init_disc_model)

	try:
		plot_losses(
			disc_losses=load_pickle(fpath=os.path.join(models_dir, f"disc_losses.gz")),
			gen_losses=load_pickle(fpath=os.path.join(models_dir, f"gen_losses.gz")),
			loss_fname=os.path.join(metrics_dir, f"losses_iteration.png"),
		)
	except Exception as e:
		print(f"<!> {e}")

	try:
		plot_losses(
			disc_losses=load_pickle(fpath=os.path.join(models_dir, f"mean_disc_losses.gz")),
			gen_losses=load_pickle(fpath=os.path.join(models_dir, f"mean_gen_losses.gz")),
			loss_fname=os.path.join(metrics_dir, f"mean_losses_epoch.png"),
		)
	except Exception as e:
		print(f"<!> {e}")

	test(
		dataloader=dataloader,
		gen=model_gen, 
		disc=model_disc,
		latent_noise_dim=opt.nz,
		device=device,
	)

if __name__ == '__main__':
	#os.system("clear")
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))