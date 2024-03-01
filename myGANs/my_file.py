import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--zeroCenteredGP', type=bool, default=False, help='zero-centered gradient penalty (def: False)')
parser.add_argument('--spectralNormGen', type=bool, default=True, help='Spectrally Normalized Generator')
parser.add_argument('--spectralNormCritic', type=bool, default=True, help='Spectrally Normalized Critic')
parser.add_argument('--spectralNormDisc', type=bool, default=False, help='Spectrally Normalized Discriminator')
opt = parser.parse_args()
print(opt)

if opt.zeroCenteredGP:
	print(f"nice, opt.zeroCenteredGP: {opt.zeroCenteredGP}")