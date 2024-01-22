import torch
import torch.nn as nn

def weights_init(m): # zero-centered Normal distribution with std 0.02.
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
		torch.nn.init.constant_(m.bias, 0)

# class Generator(nn.Module):
# 	def __init__(self, ngpu: int = 1, nz: int = 100, feature_g: int = 256, nCh: int = 3,):
# 		super(Generator, self).__init__()
# 		self.ngpu = ngpu
# 		self.main = nn.Sequential(
			
# 			nn.ConvTranspose2d(in_channels=nz, out_channels=feature_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g * 8),
# 			nn.ReLU(inplace=True),

# 			nn.ConvTranspose2d(in_channels=feature_g * 8, out_channels=feature_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g * 4),
# 			nn.ReLU(inplace=True),
			
# 			nn.ConvTranspose2d(in_channels=feature_g * 4, out_channels=feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g * 2),
# 			nn.ReLU(inplace=True),
			
# 			nn.ConvTranspose2d(in_channels=feature_g * 2, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g),
# 			nn.ReLU(inplace=True),

# 			nn.ConvTranspose2d(in_channels=feature_g, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g),
# 			nn.ReLU(inplace=True),
			
# 			nn.ConvTranspose2d(in_channels=feature_g, out_channels=feature_g, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_g),
# 			nn.ReLU(inplace=True),

# 			nn.ConvTranspose2d(in_channels=feature_g, out_channels=nCh, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.Tanh() # output normalized to [-1, 1]
# 		)

# 		# Apply spectral normalization to all convolutional layers
# 		for layer in self.main:
# 			if isinstance(layer, nn.ConvTranspose2d):
# 				nn.utils.spectral_norm(layer)

# 	def forward(self, input):
# 		if input.is_cuda and self.ngpu > 1:
# 			output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
# 		else:
# 			output = self.main(input) # [nb, ch, 256, 256]
# 		return output

# class Discriminator(nn.Module):
# 	def __init__(self, ngpu: int = 1, feature_d: int = 256, nCh: int = 3, ):
# 		super(Discriminator, self).__init__()
# 		self.ngpu = ngpu
# 		self.main = nn.Sequential(

# 			nn.Conv2d(in_channels=nCh, out_channels=feature_d, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.LeakyReLU(negative_slope=0.2, inplace=True),

# 			nn.Conv2d(in_channels=feature_d, out_channels=feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_d * 2),
# 			nn.LeakyReLU(negative_slope=0.2, inplace=True),

# 			nn.Conv2d(in_channels=feature_d * 2, out_channels=feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_d * 4),
# 			nn.LeakyReLU(negative_slope=0.2, inplace=True),

# 			nn.Conv2d(in_channels=feature_d * 4, out_channels=feature_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.BatchNorm2d(num_features=feature_d * 8),
# 			nn.LeakyReLU(negative_slope=0.2, inplace=True),

# 			nn.Conv2d(in_channels=feature_d * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
# 			nn.Sigmoid() # final probability through a Sigmoid activation function
# 		)
# 		# Apply spectral normalization to all convolutional layers
# 		for layer in self.main:
# 			if isinstance(layer, nn.Conv2d):
# 				nn.utils.spectral_norm(layer)
	
# 	def forward(self, input):
# 		if input.is_cuda and self.ngpu > 1:
# 			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
# 		else:
# 			output = self.main(input)
# 			# print(f"desc: forward: raw: {output.shape}")
# 			# print(f"desc: forward: raw.view(-1, 1): {output.view(-1, 1).shape}")
# 			# print(f"desc: forward: raw.view(-1, 1).squeeze(1): {output.view(-1, 1).squeeze(1).shape}")
# 		return output.view(-1, 1).squeeze(1) # Removes singleton dimension (dimension with size 1)

class Generator(nn.Module):
	def __init__(self, ngpu: int = 1, nz: int = 100, feature_g: int = 256, nCh: int = 3, spectral_norm: bool = False):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.spectral_norm = spectral_norm

		layers = [
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
			nn.Tanh()  # output normalized to [-1, 1]
		]

	if self.spectral_norm:
		layers = [nn.utils.spectral_norm(layer) if isinstance(layer, nn.Conv2d) else layer for layer in layers]

	self.main = nn.Sequential(*layers)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)  # [nb, ch, 256, 256]
		return output

class Discriminator(nn.Module):
	def __init__(self, ngpu: int = 1, feature_d: int = 256, nCh: int = 3, spectral_norm: bool = False):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.spectral_norm = spectral_norm

		layers = [
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
			nn.Sigmoid()  # final probability through a Sigmoid activation function
		]

		if self.spectral_norm:
			layers = [nn.utils.spectral_norm(layer) if isinstance(layer, nn.Conv2d) else layer for layer in layers]

		self.main = nn.Sequential(*layers)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output.view(-1, 1).squeeze(1)  # Removes singleton dimension (dimension with size 1)