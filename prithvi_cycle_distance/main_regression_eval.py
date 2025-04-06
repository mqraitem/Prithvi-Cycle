import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import torch.nn  as nn
import os
import wandb
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import data_path,save_checkpoint,str2bool
from prithvi_hf.prithvi import PrithviSeg
from data_load_prithvi_cycle import cycle_dataset
from matplotlib import pyplot as plt


################################## useful class and functions ##################################################

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



def compute_accuracy(labels, output, total_correct, total_labels, total_predicted):

	for idx in range(4): 

		
		predicted = output[idx,0]
		labels_t = labels[idx]

		predicted = predicted.flatten()
		labels_t = labels_t.flatten()

		#remove -1 
		mask = labels_t != -1
		predicted = predicted[mask]
		labels_t = labels_t[mask]

		correct = (predicted - labels_t).detach().cpu().numpy()

		total_correct[idx].append(correct)
		total_labels[idx].append(labels_t.detach().cpu().numpy())
		total_predicted[idx].append(predicted.detach().cpu().numpy())	

	return total_correct, total_labels, total_predicted

def eval_data_loader(data_loader,model, device):

	model.eval()
	eval_loss = 0.0

	all_out = {}
	all_mask = {}
	all_out_month = {}
	all_out_diff = {}

	eval_loss = 0.0
	with torch.no_grad():
		for j,data in tqdm(enumerate(data_loader), total=len(data_loader)):
			input = data["image"].to(device)[:, 0]
			mask = data["mask"].to(device)

			out=model(input)
			eval_loss += segmentation_loss(mask=data["mask_distance"].to(device),pred=out,device=device).item() * input.size(0)  # Multiply by batch size

			for mask_, model_output, hls_tile_n, last_month in zip(mask, out, data["hls_tile_name"], data["last_month"]):
				if hls_tile_n not in all_mask:
					all_mask[hls_tile_n] = mask_

				if hls_tile_n not in all_out:
					all_out[hls_tile_n] = []

				if hls_tile_n not in all_out_month:	
					all_out_month[hls_tile_n] = []

				if hls_tile_n not in all_out_diff:
					all_out_diff[hls_tile_n] = []


				model_output = model_output
				predicted_months = (last_month + model_output) 
				all_out_month[hls_tile_n].append(last_month)	
				all_out[hls_tile_n].append(predicted_months)
				all_out_diff[hls_tile_n].append(model_output)
			
	total_correct = {i:[] for i in range(4)}
	total_labels = {i:[] for i in range(4)}
	total_predicted = {i:[] for i in range(4)}	
	total_all_month_predicted = {i:[] for i in range(4)}

	for hls_tile_n in tqdm(all_out):
		out_hls_tile = torch.stack(all_out[hls_tile_n], dim=0)
		out_hlst_tile_diff = torch.stack(all_out_diff[hls_tile_n], dim=0)
		mask_hls_tile = all_mask[hls_tile_n]

		#take the most common month
		predicted_month = torch.mean(out_hls_tile, dim=0)
		total_correct, total_labels, total_predicted = compute_accuracy(mask_hls_tile, predicted_month, total_correct, total_labels, total_predicted)
	
		#order out_hls_tile by month
		out_hls_tile_diff_sorted = out_hlst_tile_diff[torch.argsort(torch.Tensor(all_out_month[hls_tile_n]))]
		for i in range(4):
			total_all_month_predicted[i].append(out_hls_tile_diff_sorted[:, i, 0].unsqueeze(0))

	# Compute the average accuracy for each dataset
	
	acc_dataset_val = {i:np.mean(np.abs(np.concatenate(total_correct[i]))) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)
	return epoch_loss_val, acc_dataset_val, total_correct, total_labels, total_predicted, total_all_month_predicted

#######################################################################################

def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the model")
	parser.add_argument("--freeze", type=str2bool, default=False, help="Whether to unfreeze the model or not")
	parser.add_argument("--logging", type=str2bool, default=False, help="Whether to log the results or not")
	parser.add_argument("--class_weights", type=str2bool, default=False, help="Whether to use class weights or not")
	parser.add_argument("--model_size", type=str, default="300m", help="Size of the model to use")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False, help="Whether to load a checkpoint or not")
	parser.add_argument("--group_name", type=str, default="default", help="Group name for wandb")
	args = parser.parse_args()


	wandb_config = {
		"learningrate": args.learning_rate,
		"freeze": args.freeze,
		"classweights": args.class_weights,
		"model_size": args.model_size,
		"load_checkpoint": args.load_checkpoint, 
	}

	wandb_name = "_".join([f"{k}-{v}" for k, v in wandb_config.items()])	

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	group_name = args.group_name 

	if args.logging: 
		wandb.init(
				project="Geospatial_Distance",
				group=group_name,
				config = wandb_config, 
				name=wandb_name,
				)
		wandb.run.log_code(".")

	path_train=data_path("training",config["data_dir"])
	path_val=data_path("validation",config["data_dir"])
	path_test=data_path("testing",config["data_dir"])

	cycle_dataset_train=cycle_dataset(path_train,split="train", region="Northeast")
	cycle_dataset_val=cycle_dataset(path_val,split="val", region="Northeast")
	cycle_dataset_test=cycle_dataset(path_test,split="test",  region="Northeast")

	train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=2)
	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)

	device = "cuda"
	weights_path = config["pretrained_cfg"]["prithvi_model_new_weight"] if args.load_checkpoint else None
	model=PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=1, model_size=args.model_size) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
	model=model.to(device)

	model.load_state_dict(torch.load("/usr4/cs505/mqraitem/ivc-ml/geo/Prithvi-data/Northeast/default/learningrate-0.0001_freeze-False_classweights-False_model_size-300m_load_checkpoint-False.pth")["model_state_dict"])

	model.backbone.model.encoder.eval()
	for blk in model.backbone.model.encoder.blocks:
		for param in blk.parameters():
			param.requires_grad = False
		
	checkpoint_dir = config["training"]["checkpoint_dir"] + f"/Northeast/{group_name}"
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"
	


	model_paths_to_name = { 
		"/usr4/cs505/mqraitem/ivc-ml/geo/Prithvi-data-distance/Northeast/regression_freeze-False_modelsize-300m_loadcheckpoint-True/learningrate-0.001_freeze-False_classweights-False_model_size-300m_load_checkpoint-True.pth": "Pretrained->Fine-tuned",
	}


	region = "Northeast"
	for checkpoint in model_paths_to_name: 
		model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
		epoch_loss_test, acc_dataset_test, total_correct_test, total_labels_test, total_predicted_test, total_all_month_predicted = eval_data_loader(test_dataloader, model, device)
	
		print(f"Region: {region}")
		print(f"Test avg acc: {np.mean(list(acc_dataset_test.values()))}")	

		for idx in range(4): 
			
			total_correct_test_idx = np.concatenate(total_correct_test[idx])
			total_labels_test_idx = np.concatenate(total_labels_test[idx])
			total_predicted_test_idx = np.concatenate(total_predicted_test[idx])

			print(total_labels_test_idx.shape)
			print(total_predicted_test_idx.shape)

			plt.hist(total_labels_test_idx, bins=100, alpha=0.5, label=f"Labels {idx}")
			plt.hist(total_predicted_test_idx, bins=50, alpha=0.5, label=f"Predicted {idx}")

			plt.legend()
			plt.xlabel("Months")
			plt.ylabel("Frequency")

			plt.savefig(f"plots/{region}_test_hist_{idx}.png")
			plt.clf()


			plt.hist(total_correct_test_idx, bins=100, alpha=0.5, label=f"Errors {idx}")
			plt.legend() 
			plt.xlabel("Months")
			plt.ylabel("Frequency")

			plt.savefig(f"plots/{region}_test_hist_errors_{idx}.png")
			plt.clf()


			
			total_month = torch.cat(total_all_month_predicted[idx], dim=0)
			total_month = torch.mean(total_month, dim=(0, 2, 3)).detach().cpu().numpy()

			#plot scatter plot 
			x = list(range(1, 13))	
			y = total_month			

			#plot lineplot 
			plt.plot(x, y, label=f"Predicted {idx}")
			#plot vertical line at x = np.mean(total_labels_test_idx)
			plt.axvline(x=np.mean(total_labels_test_idx), color='r', linestyle='--', label="Mean Label")
			plt.xlabel("Month (end of sequence)")
			plt.ylabel("Predicted Distance")

			plt.savefig(f"plots/{region}_test_scatter_{idx}.png")

			plt.clf()
	
if __name__ == "__main__":
	main()
