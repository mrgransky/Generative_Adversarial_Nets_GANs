from torchvision.utils import make_grid
from torchvision.utils import save_image

from typing import List, Set, Dict, Tuple
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.models as models
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
import pandas as pd
import numpy as np
import dill 
import gzip

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
# print(f"FID Score: {fid_score}")

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
	print(f"Elapsed_t: {elpt:.3f} s | {fsize_dump:.2f} MB".center(150, " "))

def plot_losses(disc_losses: List[float], gen_losses: List[float], saveDIR: str="path/to/savingDIR"):
	print(f">> Saving {len(disc_losses)} D_losses & {len(gen_losses)} G_losses...")
	
	plt.figure(figsize=(10, 6), facecolor='white')
	plt.plot(disc_losses, label='Discriminator', alpha=0.8)
	plt.plot(gen_losses, label='Generator', alpha=0.8)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Generator and Discriminator Losses over Training')
	plt.legend()
	
	plt.savefig(
		fname=os.path.join(saveDIR, "loss.png"),
		bbox_inches ="tight",
		facecolor="white", 
		edgecolor='none',
		transparent = True,
		dpi=200,
	)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
	image_tensor = (image_tensor + 1) / 2
	image_unflat = image_tensor.detach().cpu()
	image_grid = make_grid(image_unflat[:num_images], nrow=5)
	plt.imshow(image_grid.permute(1, 2, 0).squeeze())
	plt.show()