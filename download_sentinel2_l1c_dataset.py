import glob
import numpy as np
import os
import sys
import urllib.request
from natsort import natsorted

if os.path.expanduser('~') == "/home/farid":
	dataset_dir = f"{os.getenv('HOME')}/datasets" # 
elif os.path.expanduser('~') == "/users/alijanif":
	dataset_dir = "/scratch/project_2004072" # scratch folder in my puhti account!
	tmp_dir = "/scratch/project_2004072/trashes"
	# nc_files_path = os.path.join(dataset_dir, 'sentinel2-l1c-random-rgb-image')
elif os.path.expanduser('~') == "/home/ubuntu":
	dataset_dir = f"{os.getenv('HOME')}/datasets" # Pouta
	tmp_dir = os.getenv('HOME') # home dir MUST BE DELETED!
else:
	print(f"{os.path.expanduser('~')} is Unknown User!")
	sys.exit(0)

print(f">>> User: {os.path.expanduser('~')}: Loading nc files from: {dataset_dir} ...")
nc_files_path = os.path.join(dataset_dir, 'sentinel2-l1c-random-rgb-image')

if not os.path.exists(nc_files_path):
	print(f"DIR: {nc_files_path} doesn't exist, creating one...")
	os.makedirs(nc_files_path)

# go through subdirectories recursively to find files
nc_filenames = glob.glob(nc_files_path + '/**/*.nc', recursive=True)
print(type(nc_filenames), len(nc_filenames))
#print(nc_filenames)

nc_filenames_dict = {}
for nc_filename in nc_filenames:
	nc_filenames_dict[nc_filename] = True

# Download data files missing locally. Do not download files that are already in Google Drive.
# Store downloaded zip files in Google Drive for faster use later.
# The filenames are indexed here in plaintext

nc_zip_files_url = 'https://a3s.fi/swift/v1/AUTH_ef7e4759a26645089a38de9633f52afd/sentinel2-l1c-random-rgb-image-zip/'
nc_zip_files_path_google_drive = "/content/drive/MyDrive/sentinel2-l1c-random-rgb-image-zip"
nc_zip_filenames = urllib.request.urlopen(nc_zip_files_url).read().decode('utf-8').splitlines()
nc_zip_urls = list(map(lambda x: nc_zip_files_url + x, nc_zip_filenames))

print(f"nc_zip_filenames: {len(nc_zip_filenames)}".center(100, " "))
print(nc_zip_filenames)
print("nc_zip_filenames".center(100, " "))

print(f"nc_zip_urls: {len(nc_zip_urls)}".center(80, "-"))
print(nc_zip_urls)
print("nc_zip_urls".center(80, "-"))

for nc_zip_url, nc_zip_filename in zip(nc_zip_urls, nc_zip_filenames):
	(_, _, count, start) = nc_zip_filename.split("-")
	(start, _) = start.split(".")
	count = int(count)
	start = int(start)
	for i in range(start, start + count):
		if f"{nc_files_path}/{str(i).zfill(6)}.nc" not in nc_filenames_dict:
			print(f"<!> File {nc_files_path}/{str(i).zfill(6)}.nc not found (yet).")
			if not os.path.exists(f"{tmp_dir}/{nc_zip_filename}"):
				if not os.path.exists(f"{nc_zip_files_path_google_drive}/{nc_zip_filename}"):
					print(f"\n>> Downloading URL: {nc_zip_url} using wget...")
					os.system(f"wget -O {tmp_dir}/{nc_zip_filename} {nc_zip_url}")
					if os.path.exists("/content/drive/MyDrive"):
						print(f"Storing to Google Drive {nc_zip_files_path_google_drive}/{nc_zip_filename}")
						if not os.path.exists(nc_zip_files_path_google_drive):
							os.makedirs(nc_zip_files_path_google_drive)
						os.system(f"cp -v {tmp_dir}/{nc_zip_filename} {nc_zip_files_path_google_drive}/{nc_zip_filename}")
				else:
					print(f"Copying from Google Drive {nc_zip_files_path_google_drive}/{nc_zip_filename}")
					os.system(f"cp -v {nc_zip_files_path_google_drive}/{nc_zip_filename} {tmp_dir}/{nc_zip_filename}")
			print(f"Extracting {tmp_dir}/{nc_zip_filename} to {nc_files_path}")
			os.system(f"unzip {tmp_dir}/{nc_zip_filename} -d {nc_files_path}")
			# os.system(f"rm -rfv {tmp_dir}/{nc_zip_filename}")
			break
