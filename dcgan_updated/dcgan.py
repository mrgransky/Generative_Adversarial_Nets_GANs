import argparse
import os
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

# how to run:
# python dcgan.py --dataset cifar10 --dataroot .
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=os.cpu_count())
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='H/W of input image...')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

parser.add_argument('--ngf', type=int, default=64) # why?
parser.add_argument('--ndf', type=int, default=64) # why?

parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='misc', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.outf, exist_ok=True)

if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
	print("WARNING: You have mps device, to enable macOS GPU run with --mps")
	
if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
	raise ValueError(f"`dataroot` parameter is required for dataset: {opt.dataset}")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
	# folder dataset
	dataset = dset.ImageFolder(
		root=opt.dataroot,
		transform=transforms.Compose(
			[
				transforms.Resize(opt.imageSize),
				transforms.CenterCrop(opt.imageSize),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]
		)
	)
	nc=3
elif opt.dataset == 'lsun':
	classes = [ c + '_train' for c in opt.classes.split(',')]
	dataset = dset.LSUN(
		root=opt.dataroot, 
		classes=classes,
		transform=transforms.Compose(
			[
				transforms.Resize(opt.imageSize),
				transforms.CenterCrop(opt.imageSize),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]
		)
	)
	nc=3
elif opt.dataset == 'cifar10':
	dataset = dset.CIFAR10(
		root=opt.dataroot, 
		download=True,
		transform=transforms.Compose(
			[
				transforms.Resize(opt.imageSize),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]
		)
	)
	nc=3
elif opt.dataset == 'mnist':
	dataset = dset.MNIST(
		root=opt.dataroot, 
		download=True,
		transform=transforms.Compose(
			[
				transforms.Resize(opt.imageSize),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,)),
			]
		)
	)
	nc=1
elif opt.dataset == 'fake':
	dataset = dset.FakeData(
		image_size=(3, opt.imageSize, opt.imageSize),
		transform=transforms.ToTensor(),
	)
	nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(
	dataset=dataset, 
	batch_size=opt.batchSize,
	num_workers=int(opt.workers),
)
print(len(dataloader), type(dataloader), dataloader)

use_mps = opt.mps and torch.backends.mps.is_available()

if opt.cuda:
	device = torch.device("cuda:0")
elif use_mps:
	device = torch.device("mps")
else:
	device = torch.device("cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight, 1.0, 0.02)
		torch.nn.init.zeros_(m.bias)

############# Generator #############
class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)
	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG))
print(f"Generator".center(100, "-"))
print(netG)
############# Generator #############

############# Discriminator #############
class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ndf * 2),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ndf * 4),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=ndf * 8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
			nn.Sigmoid()
		)
	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
			print(f"desc: forward: raw: {output.shape}")
			print(f"desc: forward: raw.view(-1, 1): {output.view(-1, 1).shape}")
			print(f"desc: forward: raw.view(-1, 1).squeeze(1): {output.view(-1, 1).squeeze(1).shape}")
		return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(f"Discriminator".center(100, "-"))
print(netD)
############# Discriminator #############
# sys.exit(0)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
	opt.niter = 1

for epoch in range(opt.niter):
	for i, (data, _) in enumerate(dataloader): # (bidx, (img, lbl)) in typical dataloader for typical dataset, eg cifar10
		print(epoch, i, type(data), data.shape)
		############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		
		# train with real
		netD.zero_grad()
		real_cpu = data.to(device)
		batch_size = real_cpu.size(0)
		# print(batch_size)
		label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
		output = netD(real_cpu)
		print(output)
		print(label.shape, output.shape)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()
		
		# train with fake
		noise = torch.randn(batch_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		output = netD(fake.detach())
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()
		
		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost
		output = netD(fake)
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()
		# print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
		print(f"Epoch {epoch}/{opt.niter} Batch {i}/{len(dataloader)} D_loss: {errD.item():.4f} G_loss: {errG.item():.4f}")
		if i % 100 == 0:
			vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True)
			fake = netG(fixed_noise)
			vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)
		if opt.dry_run:
			break

	# do checkpointing
	torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
	torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))