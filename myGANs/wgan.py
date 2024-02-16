from utils import *
from dataloader import *
from networks import *
from test import *

##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################

# how to run:
# in Puhti:
## dcgan:
# python wgan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc --lr 0.0002 --wganMethodIdx 0 --numWorkers 8 --nepochs 50 --batchSZ 8 --dispInterval 100

## sngan:
# python wgan.py --rgbDIR /scratch/project_2004072/sentinel2-l1c_RGB_IMGs --resDIR /scratch/project_2004072/GANs/misc --lr 0.0002 --wganMethodIdx 1 --numWorkers 8 --nepochs 50 --batchSZ 4 --dispInterval 100

# in Puota:
# python wgan.py --rgbDIR /media/volume/datasets/sentinel2-l1c_RGB_IMGs --resDIR /media/volume/trash/GANs/misc --batchSZ 16 --cudaNum 3

# in Local laptop:
# python wgan.py --rgbDIR $HOME/datasets/sentinel2-l1c_RGB_IMGs --resDIR $HOME/datasets/GANs_results/misc

# WGAN-GP ref link: https://lilianweng.github.io/posts/2017-08-20-gan/

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=6, help='training epochs')
parser.add_argument('--batchSZ', type=int, default=4, help='input batch size')
parser.add_argument('--imgSZ', type=int, default=256, help='H & W input images') # can't change now!!
parser.add_argument('--imgNumCh', type=int, default=3, help='Image channel(s), def: 3 RGB')
parser.add_argument('--nz', type=int, default=100, help='noise latent z vector size')

parser.add_argument('--feature_g', type=int, default=256)
parser.add_argument('--feature_d', type=int, default=256)

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam. default=0.0')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.9')

parser.add_argument('--spectralNormGen', type=bool, default=True, help='Spectrally Normalized Generator')
parser.add_argument('--spectralNormCritic', type=bool, default=True, help='Spectrally Normalized Critic')
parser.add_argument('--spectralNormDisc', type=bool, default=False, help='Spectrally Normalized Discriminator')

parser.add_argument('--resDIR', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--rgbDIR', required=True, help='path to RGB dataset')

parser.add_argument('--numWorkers', type=int, default=16, help='number of cpu core(s)')
parser.add_argument('--nGPUs', type=int, default=1, help='number of GPU(s)') # torch.cuda.device_count()
parser.add_argument('--cudaNum', type=int, default=0, help='CUDA') # torch.cuda.device_count()
parser.add_argument('--wganMethodIdx', type=int, default=1, help='WGAN method (default: WGAN-GP)')
parser.add_argument('--dispInterval', type=int, default=500, help='Display Interval')

opt = parser.parse_args()

device = torch.device(f"cuda:{opt.cudaNum}") if torch.cuda.is_available() else torch.device("cpu")
print(f">> Running Using device: {device}")

cudnn.benchmark: bool = True

GAN_METHODs: List[str] = ["wgan", "wgan-gp"]

opt.resDIR += f"_{GAN_METHODs[opt.wganMethodIdx]}"
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

if opt.spectralNormGen:
	opt.resDIR += f"_spectralNormGen_{opt.spectralNormGen}"

if opt.spectralNormCritic:
	opt.resDIR += f"_spectralNormCritic_{opt.spectralNormCritic}"

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

NETWORKS = {
	"generator": Generator(
		nz=int(opt.nz),
		feature_g=int(opt.feature_g), 
		nCh=int(opt.imgNumCh),
		spectral_norm = opt.spectralNormGen,
	),
	"discriminator": Discriminator(
		feature_d=int(opt.feature_d), 
		nCh=int(opt.imgNumCh),
		spectral_norm = opt.spectralNormDisc,
	),
	"critic": Critic(
		feature_d=int(opt.feature_d), # feature_d = feature_critic
		nCh=int(opt.imgNumCh),
		spectral_norm = opt.spectralNormCritic,
	),
}

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

def get_network_(netName: str="generator", device: str="cuda:0"):
	print(f"Loading Network: {netName.title()}".center(120, "-"))
	net = NETWORKS.get(netName).to(device)
	get_param_(model=net)
	return net

def train(init_gen_model=None, init_critic_model=None):
	print(f"Train with {opt.nGPUs} GPU(s) {device} | {opt.numWorkers} CPU core(s)".center(100, " "))

	if init_gen_model and init_critic_model:
		gen_net = init_gen_model		
		critic_net = init_critic_model
	else:
		gen_net = get_network_(netName="generator", device=device)
		critic_net = get_network_(netName="critic", device=device)

	critic_net.apply(weights_init)
	gen_net.apply(weights_init)
	
	# loss fcn:
	criterion = torch.nn.BCELoss() # since we have sigmoid at the final layer of Critic

	# optimizers
	opt_critic = torch.optim.Adam(
		params=critic_net.parameters(), 
		lr=opt.lr, 
		betas=(opt.beta1, opt.beta2),
	)
	opt_gen = torch.optim.Adam(
		params=gen_net.parameters(), 
		lr=opt.lr, 
		betas=(opt.beta1, opt.beta2),
	)

	critic_losses = list()
	gen_losses = list()
	mean_critic_losses = list()
	mean_generator_losses = list()

	best_critic_loss = float('inf')
	best_generator_loss = float('inf')

	best_critic_state_dict = None
	best_generator_state_dict = None

	num_critics: int = 5
	LAMBDA_GP: int = 10

	for epoch in range(opt.nepochs):
		mean_critic_loss = 0
		mean_generator_loss = 0
		for batch_idx, (batch_images, batch_images_names) in enumerate(dataloader):
			# print(epoch+1, batch_idx, type(batch_images), batch_images.shape, batch_images_names)
			real_samples = batch_images.to(device)
			real_samples_names = batch_images_names # not important that much!
			cur_batch_size = real_samples.size(0) # [nb, ch, H, W] # must be double checked!

			################################################################################
			# (1) Update Critic network
			################################################################################
			mean_iteration_critic_loss = 0
			for _ in range(num_critics):
				# (1.1) Train Critic with real images
				# critic_net.zero_grad() # can't be true!!!
				opt_critic.zero_grad() # # zero parameter gradients
				critic_real_pred = critic_net(real_samples)
				
				# (1.2) Train Critic with fake generated images
				fake_noise = torch.randn(cur_batch_size, opt.nz, device=device) # [nb, 100]
				fake_samples = gen_net(fake_noise)
				critic_fake_pred = critic_net(fake_samples.detach())
				gradient_penalty = get_gradient_penalty(
					critic=critic_net,
					real_samples=real_samples,
					fake_samples=fake_samples,
					device=device,
				)
				critic_loss = get_critic_loss(
					real_pred=critic_real_pred, 
					fake_pred=critic_fake_pred, 
					GP=gradient_penalty,
					LAMBDA=LAMBDA_GP,
				)
				mean_iteration_critic_loss += (critic_loss.item() / num_critics) # tracking avg critic loss in this batch
				critic_loss.backward(retain_graph=True) # Update gradients
				opt_critic.step() # Update optimizer
			critic_losses.append(mean_iteration_critic_loss)

			##############################
			# (2) Update Generator network
			##############################

			# gen_net.zero_grad() # can't be true!
			opt_gen.zero_grad()
			# fake_noise_2 = torch.randn(cur_batch_size, opt.nz, 1, 1, device=device) # [nb, 100, 1, 1] # H&W (1x1) of generated images
			fake_noise_2 = torch.randn(cur_batch_size, opt.nz, device=device) # [nb, 100]
			fake_samples_2 = gen_net(fake_noise_2)
			critic_fake_pred = critic_net(fake_samples_2)

			gen_loss = get_generator_loss(fake_pred=critic_fake_pred)
			
			gen_loss.backward()
			opt_gen.step()
			
			mean_generator_loss += gen_loss.item()

			gen_losses.append(gen_loss.item())

			if ((batch_idx+1) % opt.dispInterval == 0) or (batch_idx+1 == len(dataloader)):
				print(
					f"Epoch {epoch+1}/{opt.nepochs} Batch {batch_idx+1}/{len(dataloader)} "
					f"Critic_loss[batch]: {critic_loss.item():.3f} Gen_loss[batch]: {gen_loss.item():.3f} "
				)
				real_batch_img_names = f"{'_'.join(real_samples_names[:-1])}_{real_samples_names[-1].split('.')[0]}"
				vutils.save_image(
					tensor=real_samples, 
					fp=os.path.join(real_imgs_dir, f"epoch{epoch+1}_batchIDX_{batch_idx+1}_Real_IMGs_{real_batch_img_names.replace('.png', '')}.png"), 
					normalize=True,
				)
				vutils.save_image(
					tensor=fake_samples.detach(), 
					fp=os.path.join(fake_imgs_dir, f"fake_samples_ep_{epoch+1}_batchIDX_{batch_idx+1}.png"), 
					normalize=True,
				)

		# save models of all epochs
		torch.save(gen_net.state_dict(), os.path.join(checkponts_dir, f"Generator_model_epoch_{epoch+1}.pth"))
		torch.save(critic_net.state_dict(), os.path.join(checkponts_dir, f"Critic_model_epoch_{epoch+1}.pth"))

		# Calculate mean losses at the end of the epoch
		mean_critic_loss /= len(dataloader)
		mean_generator_loss /= len(dataloader)

		# Check if the current model has lower losses than the best
		if mean_critic_loss < best_critic_loss:
			print(
				f"Found better Critic model @ epoch: {epoch+1} "
				f"prev_best_loss: {best_critic_loss} "
				f"curr_loss: {mean_critic_loss:.3f}".center(150, " ")
			)
			best_critic_loss = mean_critic_loss
			best_disc_model = critic_net
			best_critic_state_dict = critic_net.state_dict()

		if mean_generator_loss < best_generator_loss:
			print(
				f"Found better Generator model @ epoch: {epoch+1} "
				f"prev_best_loss: {best_generator_loss} "
				f"curr_loss: {mean_generator_loss:.3f}".center(150, " ")
			)
			best_generator_loss = mean_generator_loss
			best_gen_model = gen_net
			best_generator_state_dict = gen_net.state_dict()
		
		print(f"\tEpoch {epoch+1}/{opt.nepochs} < Mean > Critic_loss: {mean_critic_loss:.3f} Gen_loss: {mean_generator_loss:.3f}")

		mean_critic_losses.append(mean_critic_loss)
		mean_generator_losses.append(mean_generator_loss)

	# Save the best models after training is complete
	torch.save(best_generator_state_dict, os.path.join(checkponts_dir, f"Generator_model_best.pth"))
	torch.save(best_critic_state_dict, os.path.join(checkponts_dir, f"Critic_model_best.pth"))

	save_pickle(
		pkl=critic_losses,
		fname=os.path.join(models_dir, f"critic_losses.gz"),
	)

	save_pickle(
		pkl=gen_losses,
		fname=os.path.join(models_dir, f"gen_losses.gz"),
	)

	save_pickle(
		pkl=mean_critic_losses,
		fname=os.path.join(models_dir, f"mean_critic_losses.gz"),
	)

	save_pickle(
		pkl=mean_generator_losses,
		fname=os.path.join(models_dir, f"mean_gen_losses.gz"),
	)

	return best_gen_model, best_disc_model

def main():
	init_gen_model = get_network_(netName="generator", device=device)
	init_critic_model = get_network_(netName="critic", device=device)
	try:
		model_gen = init_gen_model
		model_critic = init_critic_model
		model_gen.load_state_dict(torch.load(os.path.join(checkponts_dir, f"Generator_model_best.pth")))
		model_critic.load_state_dict(torch.load(os.path.join(checkponts_dir, f"Critic_model_best.pth")))
		print("Loaded best generator and critic models successfully.")
	except Exception as e:
		print(f"<!>\n{e}")
		model_gen, model_critic = train(init_gen_model, init_critic_model)

	try:
		plot_losses(
			disc_losses=load_pickle(fpath=os.path.join(models_dir, f"critic_losses.gz")),
			gen_losses=load_pickle(fpath=os.path.join(models_dir, f"gen_losses.gz")),
			loss_fname=os.path.join(metrics_dir, f"losses_iteration.png"),
		)
	except Exception as e:
		print(f"<!>\n{e}")

	try:
		plot_losses(
			disc_losses=load_pickle(fpath=os.path.join(models_dir, f"mean_critic_losses.gz")),
			gen_losses=load_pickle(fpath=os.path.join(models_dir, f"mean_gen_losses.gz")),
			loss_fname=os.path.join(metrics_dir, f"mean_losses_epoch.png"),
		)
	except Exception as e:
		print(f"<!> \n{e}")

	test(
		dataloader=dataloader,
		gen=model_gen, 
		disc=model_critic,
		latent_noise_dim=opt.nz,
		device=device,
		nGPUs=opt.nGPUs,
		numWorkers=opt.numWorkers,
	)

if __name__ == '__main__':
	#os.system("clear")
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))