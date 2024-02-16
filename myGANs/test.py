from utils import *

def test(dataloader, gen, disc, latent_noise_dim: int=100, device: str="cuda", nGPUs: int=1, numWorkers: int=8):
	print(f"Test with {device}, {nGPUs} GPU(s) & {numWorkers} CPU core(s)".center(100, " "))
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