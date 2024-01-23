from torchvision.utils import make_grid
from torchvision.utils import save_image

from typing import List, Set, Dict, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_fid(real_images, generated_images):
	# Extract feature vectors
	real_features = extract_features(real_images)
	generated_features = extract_features(generated_images)

	# Calculate mean and covariance of feature vectors
	real_mean, real_cov = np.mean(real_features, axis=0), np.cov(real_features.T)
	generated_mean, generated_cov = np.mean(generated_features, axis=0), np.cov(generated_features.T)

	# Calculate the Frechet Inception Distance (Multivariate)
	fid_distance = np.linalg.norm((real_mean - generated_mean).astype(np.float32), ord='fro') ** 2 + np.trace(real_cov * generated_cov)

	return fid_distance

def extract_features(images):
	# Load Inception V3 model
	model = torchvision.models.inception_v3(pretrained=True)
	model.eval()

	# Extract feature vectors
	features = torch.FloatTensor()
	for image in images:
		image = image.unsqueeze(0)
		with torch.no_grad():
			output = model(image)
			feature = output.view(output.size(1), -1)
			features = torch.cat((features, feature), 0)

	return features

def plot_losses(disc_losses: List[float], gen_losses: List[float], saveDIR: str="path/to/savingDIR"):
	# # Lists to store losses for plotting

	# Training loop
	for epoch in range(opt.nepochs):
			for batch_idx, batch_images in enumerate(dataloader):
					# ... (your training code)

					# Append losses to lists
					disc_losses.append(disc_loss.item())
					gen_losses.append(gen_loss.item())

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(disc_losses, label='Discriminator Loss', alpha=0.7)
	plt.plot(gen_losses, label='Generator Loss', alpha=0.7)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Generator and Discriminator Losses over Training')
	plt.legend()
	# plt.show()
	plt.savefig(
		fname=os.path.join(saveDIR, "squares1.png"),
		bbox_inches ="tight",
		pad_inches = 1,
		transparent = True,
		# facecolor ="g",
		# edgecolor ='w',
		# orientation ='landscape'
	)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
	image_tensor = (image_tensor + 1) / 2
	image_unflat = image_tensor.detach().cpu()
	image_grid = make_grid(image_unflat[:num_images], nrow=5)
	plt.imshow(image_grid.permute(1, 2, 0).squeeze())
	plt.show()