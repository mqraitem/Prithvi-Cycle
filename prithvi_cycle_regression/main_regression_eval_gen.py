import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import torch.nn  as nn
import os
import wandb
import argparse
from PIL import Image
import pickle
from tqdm import tqdm

from utils import str2bool, data_path
from data_load_prithvi_cycle import load_raster,preprocess_image,cycle_dataset
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from prithvi_hf.prithvi import PrithviSeg
from data_load_prithvi_cycle import cycle_dataset

################################## useful class and functions ##################################################

# ['Southeast', 'Mex_Drylands', 'West', 'North_Pacific', 'Central_America', 'Northeast', 'Boreal', 'Great_Plains', 'Alaska'] 



def segmentation_loss(mask, pred, device, class_weights=None, ignore_index=-1):
	mask = mask.float()  # Convert mask to float for regression loss

	if class_weights is not None:
		criterion = nn.MSELoss(reduction="none").to(device)  # Sum-based MSELoss for proper mean calculation
	else: 
		criterion = nn.MSELoss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]  # Number of output channels
	total_valid_pixels = 0  # Counter for valid pixels

	for idx in range(num_channels):
		# Get valid mask (excluding ignore_index)
		valid_mask = mask[:, idx] != ignore_index
		if class_weights is not None:
			class_weights_channel = class_weights[:, idx]
			class_weights_channel = class_weights_channel[valid_mask].to(device)

		if valid_mask.sum() > 0:  # Ensure there are valid pixels to compute loss
			valid_pred = pred[:, idx][:, 0][valid_mask]  # Apply mask to predictions
			valid_target = mask[:, idx][valid_mask]  # Apply mask to ground truth
			
			if class_weights is not None:
				loss_ = criterion(valid_pred, valid_target) * class_weights_channel 
				loss_ = loss_.sum()
				loss += loss_

			else: 
				loss += criterion(valid_pred, valid_target)

			total_valid_pixels += valid_mask.sum().item()

	# Normalize by total valid pixels to avoid division by zero
	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)


def compute_accuracy(labels, output, total_correct):

	for idx in range(4): 

		predicted = output[:, idx][:, 0]
		predicted = torch.round(predicted)
		labels_t = labels[:, idx]

		predicted = predicted.flatten()
		labels_t = labels_t.flatten()

		#remove -1 
		mask = labels_t != -1
		predicted = predicted[mask]
		labels_t = labels_t[mask]

		correct = torch.abs(predicted - labels_t).detach().cpu().numpy()

		total_correct[idx].append(correct)

	return total_correct



#######################################################################################

def eval_data_loader(data_loader,model, device):

	model.eval()
	eval_loss = 0.0
	total_correct = {i:[] for i in range(4)}

	with torch.no_grad():
		for j,(input,mask, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

			input=input.to(device)[:, 0]
			mask=mask.to(device)
		
			out=model(input)


			loss=segmentation_loss(mask,out,device,None)
			total_correct = compute_accuracy(mask, out, total_correct)

			eval_loss += loss.item() * input.size(0) 


	acc_dataset_val = {i:np.mean(np.concatenate(total_correct[i])) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)
	return epoch_loss_val, acc_dataset_val


def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the model")
	parser.add_argument("--freeze", type=str2bool, default=False, help="Whether to unfreeze the model or not")
	parser.add_argument("--logging", type=str2bool, default=False, help="Whether to log the results or not")
	parser.add_argument("--class_weights", type=str2bool, default=False, help="Whether to use class weights or not")
	#model_size
	parser.add_argument("--model_size", type=str, default="600m", help="Model size to use")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False, help="Whether to load checkpoint or not")
	args = parser.parse_args()

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)


	to_plot_heatmap = { 
		"region":[],
		"acc": [],
		"model": [] 
	}

	model_paths_to_name = { 
		None: "Random Init",
		"/usr4/cs505/mqraitem/ivc-ml/geo/Prithvi-data-reg/Northeast/regression_freeze-False_modelsize-600m_loadcheckpoint-True/learningrate-0.001_freeze-False_classweights-False_model_size-600m_load_checkpoint-True.pth": "Pretrained->Fine-tuned",
		"/usr4/cs505/mqraitem/ivc-ml/geo/Prithvi-data-reg/Northeast/regression_freeze-False_modelsize-600m_loadcheckpoint-False/learningrate-0.001_freeze-False_classweights-False_model_size-600m_load_checkpoint-False.pth": "Random Init->Fine-tuned",
	}


	# name = "heatmap_all_similar.png"
	# regions = ["Southeast", "West", "Boreal", "Great_Plains"]

	name = "heatmap_all_disimilar.png"
	regions = ["Mex_Drylands", "North_Pacific", "Central_America", "Alaska"]

	for checkpoint in model_paths_to_name:
		all_acc_checkpoint = []
		for region in regions: 

			path_val=data_path("validation",config["data_dir"],region)
			path_test=data_path("testing",config["data_dir"],region)
			
			cycle_dataset_val=cycle_dataset(path_val,split="val")
			cycle_dataset_test=cycle_dataset(path_test,split="test")


			val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)
			test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)


			device = "cuda"
			weights_path =  None
			model=PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=1, model_size=args.model_size) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
			model=model.to(device)

			if checkpoint: 
				model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

			epoch_loss_val, acc_dataset_val = eval_data_loader(val_dataloader, model, device)
			epoch_loss_test, acc_dataset_test = eval_data_loader(test_dataloader, model, device)

			print(f"Region: {region}")
			print(f"Validation avg acc: {np.mean(list(acc_dataset_val.values()))}")
			print(f"Test avg acc: {np.mean(list(acc_dataset_test.values()))}")	
		

			to_plot_heatmap["region"].append(region)
			to_plot_heatmap["acc"].append(np.mean(list(acc_dataset_test.values())))
			to_plot_heatmap["model"].append(model_paths_to_name[checkpoint])
			all_acc_checkpoint.append(np.mean(list(acc_dataset_test.values())))
		
		#add average
		to_plot_heatmap["region"].append("Average")
		to_plot_heatmap["acc"].append(np.mean(all_acc_checkpoint))
		to_plot_heatmap["model"].append(model_paths_to_name[checkpoint])

	df = pd.DataFrame(to_plot_heatmap)

	list_order = regions + ["Average"]
	df["region"] = pd.Categorical(df["region"], categories=list_order, ordered=True)
	df = df.sort_values("region")

	heatmap_data = df.pivot(index="region", columns="model", values="acc")

	# Plot heatmap
	plt.figure(figsize=(8, 6))
	sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
	plt.xticks(rotation=0)

	plt.savefig(f"plots/{name}", bbox_inches='tight')

if __name__ == "__main__":
	main()
