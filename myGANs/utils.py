from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple
import os

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