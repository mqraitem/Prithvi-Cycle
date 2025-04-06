import rasterio
import os
from tqdm import tqdm
import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


NO_DATA = -9999
NO_DATA_FLOAT = 0.0001

def load_raster(path,if_img,crop=None):
		with rasterio.open(path) as src:
			img = src.read()

			img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
			
			#crops only lower right corner
			if crop:
				img = img[:, -crop[0]:, -crop[1]:]
		return img

def day_of_year_to_week_of_year(day_of_year):

	month_of_year = np.zeros_like(day_of_year)
	month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
	cumulative_days = np.cumsum(month_days)

	# Assign each day_of_year to a corresponding month
	for i, last_day in enumerate(cumulative_days):
		month_of_year[(day_of_year > (last_day - month_days[i])) & (day_of_year <= last_day)] = i  # Start from 0

	# Only update valid days
	month_of_year[day_of_year == -1] = -1

	return month_of_year

def day_of_year_to_decimal_month(day_of_year):
	decimal_month = np.zeros_like(day_of_year, dtype=float)
	
	# Days in each month for a non-leap year
	month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
	cumulative_days = np.cumsum(month_days)
	start_days = np.insert(cumulative_days[:-1], 0, 0)  # start of each month
	
	for i in range(12):
		# Get days that fall in the i-th month
		mask = (day_of_year > start_days[i]) & (day_of_year <= cumulative_days[i])
		days_into_month = day_of_year[mask] - start_days[i]
		decimal_month[mask] = (i + 1) + (days_into_month - 1) / month_days[i]  # -1 so day 1 is .0

	# Handle invalid days
	decimal_month[day_of_year == -1] = -1

	return decimal_month

def preprocess_image(image,means,stds):
		
		# normalize image

		number_of_channels = image.shape[0]	
		number_of_time_steps = image.shape[1]

		means1 = means.reshape(number_of_channels,number_of_time_steps,1,1)  # Mean across height and width, for each channel
		stds1 = stds.reshape(number_of_channels,number_of_time_steps,1,1)    # Std deviation across height and width, for each channel
		normalized = image.copy()
		normalized = ((image - means1) / stds1)
		#print("normalized image",normalized)
		
		normalized = torch.from_numpy(normalized.reshape(1, number_of_channels, number_of_time_steps, *normalized.shape[-2:])).to(torch.float32)
		return normalized

class cycle_dataset(Dataset):
	def __init__(self,path,split):
		self.data_dir=path
		self.split=split

		self.total_below_0 = 0 
		self.total_above_365 = 0
		self.total_nan = 0
		self.total = 0

		self.get_means_stds()

	def get_means_stds(self):
		"""
		Compute or load the mean and standard deviation for image data.
		Uses Welford's online algorithm for numerical stability.
		"""

		means_stds_path = "/usr4/cs505/mqraitem/ivc-ml/geo/data/Northeast/means_stds_regression.pkl"

		# Load precomputed means and stds
		if os.path.exists(means_stds_path):
			with open(means_stds_path, 'rb') as f:
				self.means, self.stds = pickle.load(f)
			return

		elif self.split == "test":
			raise ValueError("Cannot compute mean and std for test split")

		num_samples = 0  # Total images processed
		mean_accumulator = np.zeros((6, 4), dtype=np.float64)
		M2_accumulator = np.zeros((6, 4), dtype=np.float64)  # For variance computation
		valid_pixel_count = np.zeros((6, 4), dtype=np.float64)  # To track non-masked pixel count

		for i in tqdm(range(len(self.data_dir))):
			image_path = self.data_dir[i][0]
			mask_path = self.data_dir[i][1]  # Assuming mask path is stored in self.data_dir

			images = []
			for path in image_path:
				if_image = 1
				images.append(load_raster(path, if_image, crop=(224, 224))[:, np.newaxis])

			img = np.concatenate(images, axis=1)  # Shape: (6, 4, 224, 224)

			# Load and process mask
			final_mask = load_raster(mask_path, 1, crop=(224, 224))  # Shape: (4, 224, 224)
			mask = np.isnan(final_mask)  # Boolean mask where True indicates NaNs

			# Expand mask to match `img` shape (6, 4, 224, 224)
			expanded_mask = np.repeat(mask[np.newaxis, :, :, :], 6, axis=0)

			# Replace masked values with 0 in `img`
			img[expanded_mask] = 0

			# Compute per-channel mean only over valid (non-masked) pixels
			valid_pixels = (~expanded_mask).sum(axis=(2, 3))  # Count non-masked pixels per (6, 4) channel
			valid_pixel_count += valid_pixels

			img_mean = np.sum(img, axis=(2, 3)) / np.maximum(valid_pixels, 1)  # Avoid division by zero

			# Online mean update (Welford's Algorithm)
			num_samples += 1
			delta = img_mean - mean_accumulator
			mean_accumulator += delta / num_samples
			M2_accumulator += delta * (img_mean - mean_accumulator)  # Accumulate squared differences

		# Ensure no division by zero
		if num_samples > 0:
			means = mean_accumulator
			stds = np.sqrt(M2_accumulator / np.maximum(valid_pixel_count, 1))  # Normalize with valid pixels
		else:
			raise ValueError("No samples processed, cannot compute means and stds.")

		# Save computed means and stds
		with open(means_stds_path, 'wb') as f:
			pickle.dump([means, stds], f)

		self.means = means
		self.stds = stds

	def __len__(self):
		#print("dataset length",len(self.input_plus_mask_path))
		return len(self.data_dir)

	
	def fix_final_mask_decimal(self,final_mask):
		mask = np.isnan(final_mask)
		final_mask = np.where(mask, -1, final_mask)
		final_mask = np.where(final_mask < 0, 0, final_mask)
		final_mask = np.where(final_mask > 365, 365, final_mask)
		final_mask = day_of_year_to_decimal_month(final_mask)
		final_mask = np.where(mask, -1, final_mask)
		return final_mask
	
	def __getitem__(self,idx):
		
		image_path=self.data_dir[idx][0]
		mask_path=self.data_dir[idx][1]
		#print("image path:",image_path)

		images = []
		for path in image_path: 
			if_image=1
			images.append(load_raster(path,if_image,crop=(224, 224))[:, np.newaxis])

		image = np.concatenate(images, axis=1)
		final_image=preprocess_image(image,self.means,self.stds)

		if_image=0
		final_mask=load_raster(mask_path,if_image,crop=(224, 224))
		final_mask = self.fix_final_mask_decimal(final_mask)

		class_weights = np.zeros_like(final_mask)
		return final_image,final_mask, class_weights
	


