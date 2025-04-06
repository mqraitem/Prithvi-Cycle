import os
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def filter_to_last_month(data_paths, data_dir, region="Northeast"):

	with open(f'{data_dir}/HLS_composites/{region}/selected_months.csv', 'r') as f:
		months = f.readlines()

	last_month = int(months[-1])
	
	data_paths = [(x, y, z, f) for (x,y,z,f) in data_paths if (z+1) == last_month]
	return data_paths

def data_path(mode, data_dir, region="Northeast"): 

	checkpoint_data = f"/usr4/cs505/mqraitem/ivc-ml/geo/data/{region}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)
		return data_dir

	hls_path = f"{data_dir}/HLS_composites/{region}"
	lsp_path = f"{data_dir}/LSP_vars/{region}"

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = [x for x in os.listdir(lsp_path) if x.endswith('.tif')]

	title_hls = ['_'.join(x.split('_')[3:5]) for x in hls_tiles]
	title_hls = list(set(title_hls))

	hls_tiles_time = [] 
	lsp_tiles_time = []
	hls_last_month = []	
	hls_tile_name = []


	hls_tile_to_num_months = {}
	for month in range(1, 13):
		past_months = [(month - i) if (month - i) > 0 else 1 for i in range(3, -1, -1)]
		timesteps = [f"2018-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			temp_ordered = [] 
			for timestep in timesteps:
				if f"HLS_composite_{timestep}_{hls_tile}" in hls_tiles:
					temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile}")

			if len(temp_ordered) != len(timesteps):
				continue
		
			temp_lsp = f"{lsp_path}/{hls_tile}" if hls_tile in lsp_tiles else None
			if len(temp_ordered) == len(timesteps) and temp_lsp:
				hls_tiles_time.append(temp_ordered)
				lsp_tiles_time.append(temp_lsp)
				hls_last_month.append(month - 1) #start from zero
				hls_tile_name.append(hls_tile)

				if hls_tile not in hls_tile_to_num_months:
					hls_tile_to_num_months[hls_tile] = []
				hls_tile_to_num_months[hls_tile].append(month - 1)

			else: 
				print(f"Missing data for {hls_tile} at {timesteps}")


	final_chosen_hls_tiles = []	
	for hls_tile_name_ in hls_tile_to_num_months:
		if len(hls_tile_to_num_months[hls_tile_name_]) == 12:
			final_chosen_hls_tiles.append(hls_tile_name_)

	hls_tiles_train = final_chosen_hls_tiles[:int(0.7*len(final_chosen_hls_tiles))]
	hls_tiles_val = final_chosen_hls_tiles[int(0.7*len(final_chosen_hls_tiles)):int(0.8*len(final_chosen_hls_tiles))]
	hls_tiles_test = final_chosen_hls_tiles[int(0.8*len(final_chosen_hls_tiles)):]

	data_dir_train = [(x, y, z, f) for (x,y,z,f) in zip(hls_tiles_time, lsp_tiles_time, hls_last_month, hls_tile_name) if f in hls_tiles_train]
	data_dir_val = [(x, y, z, f) for (x,y,z,f) in zip(hls_tiles_time, lsp_tiles_time, hls_last_month, hls_tile_name) if f in hls_tiles_val]
	data_dir_test = [(x, y, z, f) for (x,y,z,f) in zip(hls_tiles_time, lsp_tiles_time, hls_last_month, hls_tile_name) if f in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)
	
	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)
	
	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':	
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	else:
		return data_dir_test


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):

	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'train_loss': train_loss,
		'val_loss':val_loss
	}
	torch.save(checkpoint, filename)
	print(f"Checkpoint saved at {filename}")
	