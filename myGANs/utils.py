import os
import time
import dill 
import gzip
import scipy
import datetime
import argparse
import sys
import random

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.io import read_image
from typing import List, Set, Dict, Tuple
import torch.nn.functional as Fun
import torchvision.models as models

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

def get_critic_loss(real_pred, fake_pred, GP, LAMBDA: int=10):
	critic_loss = torch.mean(fake_pred) - torch.mean(real_pred) + (GP * LAMBDA)
	return critic_loss

def get_generator_loss(fake_pred):
	gen_loss = -1. * torch.mean(fake_pred)
	return gen_loss

def get_gradient_penalty(critic, real_samples, fake_samples, zero_centered: bool=False, device: str="cuda:0"):
	# Mix real_samples and fake_samples randomly between them.
	alpha = torch.rand(len(real_samples), 1, 1, 1, device=device, requires_grad=True)
	interpolated_samples = (real_samples * alpha) + (fake_samples * (1 - alpha))

	# Forward pass through critic to calculate critic scores
	interpolated_scores = critic(interpolated_samples)

	# Take the gradient of the scores with respect to the images
	gradient = torch.autograd.grad(
		inputs=interpolated_samples,
		outputs=interpolated_scores,
		grad_outputs=torch.ones_like(interpolated_scores, device=device),
		create_graph=True, # backpropagating through computational graph later in training
		retain_graph=True,
		only_inputs=True, # gradients are only calculated for interpolated samples, not critic parameters
	)[0]
	batch_size = gradient.size(0)
	gradient = gradient.view(batch_size, -1)

	if zero_centered:
		# Calculate the mean gradient
		mean_gradient = torch.mean(gradient, dim=0)

		# Calculate the Zero-Centered Gradient Penalty
		centered_gradient = gradient - mean_gradient.unsqueeze(0)

		# replace centered_gradient with gradient
		gradient = centered_gradient

	gradient_norm = gradient.norm(2, dim=1) # L2-norm of gradients along dim=1, returns tensor with one value per sample
	gradient_penalty = torch.mean((gradient_norm - 1)**2)
	
	return gradient_penalty
		
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
	real_features_list = list()
	fake_features_list = list()
	with torch.no_grad():
		for batch_idx, (batch_images, batch_images_names) in enumerate(dataloader):
			# print(batch_idx)
			real_samples = batch_images
			# print(real_samples.shape, type(real_samples), real_samples.dtype, real_samples.device)
			real_samples = Fun.interpolate(input=real_samples, size=(299, 299), mode='bilinear', align_corners=False)
			# print(f">> Entering {real_samples.shape} into inception_v3 model...")
			# print(model_inception_v3(real_samples.to(device)).requires_grad)
			real_features = model_inception_v3(real_samples.to(device)).detach().to('cpu')
			# print(type(real_features), real_features.dtype, real_features.shape, real_samples.device)
			real_features_list.append(real_features)

			# Generator to genrate fake_samples and fake_features:
			# fake_noise = torch.randn(len(batch_images), nz, 1, 1, device=device) # [nb x nz, 1, 1]
			fake_noise = torch.randn(len(batch_images), nz, device=device) # [nb x nz]			
			fake_samples = model_generator(fake_noise) # torch.Size([nb, nch, feature_g, feature_g])
			fake_samples = Fun.interpolate(fake_samples, size=(299, 299), mode='bilinear', align_corners=False)
			# print(fake_samples.shape, type(fake_samples), fake_samples.dtype)
			fake_features = model_inception_v3(fake_samples.to(device)).detach().to('cpu')
			# print(type(fake_features), fake_features.dtype, fake_features.shape)
			fake_features_list.append(fake_features)

			# print()
	# print(len(real_features_list), len(fake_features_list))
	return torch.cat(tensors=real_features_list, dim=0), torch.cat(tensors=fake_features_list, dim=0)

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

def load_pickle(fpath:str="unknown",):
	print(f"Checking for existence? {fpath}")
	st_t = time.time()
	try:
		with gzip.open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except gzip.BadGzipFile as ee:
		print(f"<!> {ee} gzip.open NOT functional => traditional openning...")
		with open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except Exception as e:
		print(f"<<!>> {e} pandas read_pkl...")
		pkl = pd.read_pickle(fpath)
	elpt = time.time()-st_t
	fsize = os.stat( fpath ).st_size / 1e6
	print(f"Loaded in: {elpt:.3f} s | {type(pkl)} | {fsize:.2f} MB".center(130, " "))
	return pkl

def plot_losses(disc_losses: List[float], gen_losses: List[float], loss_fname: str="path/to/savingDIR/loss.png"):
	print(f'>> Plotting {loss_fname.split("/")[-1].replace(".png", "")} Disc: {len(disc_losses)} | Gen: {len(gen_losses)} ...')
	
	plt.figure(figsize=(10, 4), facecolor='white')
	plt.plot(disc_losses, label='Disc', alpha=0.85, color="blue")
	plt.plot(gen_losses, label='Gen', alpha=0.4, color="red")
	plt.xlabel(loss_fname.split("/")[-1].replace(".png", "").split("_")[-1].title()) # Iteration / Epoch
	plt.ylabel(" ".join(loss_fname.split("/")[-1].replace(".png", "").split("_")[:-1]).title()) # Losses / Mean Losses
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
