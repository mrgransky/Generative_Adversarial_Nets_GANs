import os
import time
import dill 
import gzip
import scipy

import torch
import datetime
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.io import read_image
from typing import List, Set, Dict, Tuple
import torchvision.transforms.functional as F
import torchvision.models as models
from torch.nn.functional import adaptive_avg_pool2d

from scipy.linalg import sqrtm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

def get_param_(model):
	print(
		f"model: {model.__class__.__name__ } "
		f"conatins: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
	)
	print(model)
	# print()
	# for name, param in model.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.data.shape)
	# 		print(param.data)
	# 		print('-'*100)

def matrix_sqrt(x):
	y = x.cpu().detach().numpy()
	y = scipy.linalg.sqrtm(y)
	return torch.Tensor(y.real, device=x.device)

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
	return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))

def get_covariance(features):
	return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

def get_real_fake_features(dataloader, model_generator, model_inception_v3, nz: int = 100, device: str="cuda"):
	real_features_list = []
	fake_features_list = []
	with torch.no_grad():
		for batch_idx, (batch_imgs, _) in enumerate(dataloader):
			#print(batch_idx, batch_imgs.shape, type(batch_imgs), batch_imgs.dtype)
			print(batch_idx)
			real_samples = batch_imgs
			print(real_samples.shape, type(real_samples), real_samples.dtype)
			real_features = model_inception_v3(real_samples.to(device)).detach().to('cpu')
			print(type(real_features), real_features.dtype, real_features.shape)
			real_features_list.append(real_features)
			# Generator to genrate fake_samples and fake_features:
			fake_noise = torch.randn(len(batch_imgs), nz, device=device) # nb x nz
			fake_samples = model_generator(fake_noise) # torch.Size([nb, nch, feature_g, feature_g])
			fake_samples = Fun.interpolate(fake_samples, size=(299, 299), mode='bilinear', align_corners=False)
			print(fake_samples.shape, type(fake_samples), fake_samples.dtype)
			fake_features = model_inception_v3(fake_samples.to(device)).detach().to('cpu')
			print(type(fake_features), fake_features.dtype, fake_features.shape)
			fake_features_list.append(fake_features)
			print()
	print(len(real_features_list), len(fake_features_list))
	return torch.cat(tensors=real_features_list, dim=0), torch.cat(tensors=fake_features_list, dim=0)




def calculate_fid(real_images, generated_images, inception_model, batch_size=64, device='cuda'):
	def get_activations(images, model, batch_size=64, device='cuda'):
		model.eval()
		num_images = images.size(0)
		activations = torch.zeros((num_images, 2048), dtype=torch.float32, device=device)
		with torch.no_grad():
			for i in range(0, num_images, batch_size):
				batch = images[i:i + batch_size].to(device)
				pred = model(batch)[0]
				activations[i:i + batch_size] = adaptive_avg_pool2d(pred, (1, 1)).squeeze(3).squeeze(2)
		return activations

	real_activations = get_activations(real_images, inception_model, batch_size, device)
	fake_activations = get_activations(generated_images, inception_model, batch_size, device)

	# Calculate mean and covariance of activations
	mu_real, sigma_real = real_activations.mean(dim=0), torch_cov(real_activations, rowvar=False)
	mu_fake, sigma_fake = fake_activations.mean(dim=0), torch_cov(fake_activations, rowvar=False)

	# Calculate FID
	diff = mu_real - mu_fake
	covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
	if not np.isfinite(covmean).all():
		cov_corr = torch.trace(sigma_real) + torch.trace(sigma_fake) - 2 * torch.trace(sqrtm(sigma_real @ sigma_fake))
		covmean = covmean + torch.eye(covmean.shape[0], device=device) * cov_corr

	fid = diff @ diff + torch.trace(sigma_real) + torch.trace(sigma_fake) - 2 * torch.trace(covmean)
	return fid.item()

def torch_cov(m, rowvar=False):
	if m.dim() > 2:
		raise ValueError('m has more than 2 dimensions')
	if m.dim() < 2:
		m = m.view(1, -1)
	if not rowvar and m.size(0) != 1:
		m = m.t()
	fact = 1.0 / (m.size(1) - 1)
	m -= torch.mean(m, dim=1, keepdim=True)
	mt = m.t()
	return fact * m @ mt

# # Example usage:
# inception_model = models.inception_v3(pretrained=True, aux_logits=False).to(device)
# fid_score = calculate_fid(real_images, generated_images, inception_model, batch_size=64, device=device)

def save_pickle(pkl, fname:str=""):
	print(f"Saving {type(pkl)} {fname}")
	st_t = time.time()
	if isinstance(pkl, ( pd.DataFrame, pd.Series ) ):
		pkl.to_pickle(path=fname)
	else:
		# with open(fname , mode="wb") as f:
		with gzip.open(fname , mode="wb") as f:
			dill.dump(pkl, f)
	elpt = time.time()-st_t
	fsize_dump = os.stat( fname ).st_size / 1e6
	print(f"Elapsed_t: {elpt:.3f} s | {fsize_dump:.4f} MB".center(100, " "))

def plot_losses(disc_losses: List[float], gen_losses: List[float], loss_fname: str="path/to/savingDIR/loss.png"):
	print(f'>> Saving {loss_fname.split("/")[-1].replace(".png", "")}: Disc: {len(disc_losses)} | Gen: {len(gen_losses)} ...')
	
	plt.figure(figsize=(10, 4), facecolor='white')
	plt.plot(disc_losses, label='Disc', alpha=0.85, color="blue")
	plt.plot(gen_losses, label='Gen', alpha=0.4, color="red")
	plt.xlabel('Iteration')
	plt.ylabel(f'{loss_fname.split("/")[-1].replace(".png", "")}')
	plt.title(f'Generator & Discriminator {loss_fname.split("/")[-1].replace(".png", "")}')
	plt.legend(ncol=2, frameon=False)
	
	plt.savefig(
		fname=loss_fname,
		bbox_inches ="tight",
		facecolor="white",
		edgecolor='none',
		transparent = True,
		dpi=150,
	)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
	image_tensor = (image_tensor + 1) / 2
	image_unflat = image_tensor.detach().cpu()
	image_grid = make_grid(image_unflat[:num_images], nrow=5)
	plt.imshow(image_grid.permute(1, 2, 0).squeeze())
	plt.show()

def visualize(dataloader):
	sample_batch_images, sample_batch_image_names = next(iter(dataloader))
	# print(sample_batch_images.size(), sample_batch_image_names, len(sample_batch_images))
	batch_img_list = [np.transpose(sample_batch_images[bi], (1,2,0)) for bi in range( len( sample_batch_images ) ) ]
	fig, axs = plt.subplots(ncols=len(batch_img_list), squeeze=False, figsize=(8,3))
	for i, img in enumerate(batch_img_list):
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		axs[0, i].set_title(f"{sample_batch_image_names[i]}")
	plt.show()
