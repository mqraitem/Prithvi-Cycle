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


import sys
sys.path.append("../")
from utils import data_path,save_checkpoint,str2bool,filter_to_last_month
from prithvi_hf.prithvi import PrithviSeg
from data_load_prithvi_cycle import cycle_dataset

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


def compute_accuracy(labels, output, total_correct):

	for idx in range(4): 

		predicted = output[:, idx][:, 0]
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

#######################################################################################

def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the model")
	parser.add_argument("--freeze", type=str2bool, default=False, help="Whether to unfreeze the model or not")
	parser.add_argument("--logging", type=str2bool, default=False, help="Whether to log the results or not")
	parser.add_argument("--class_weights", type=str2bool, default=False, help="Whether to use class weights or not")
	parser.add_argument("--model_size", type=str, default="300m", help="Size of the model to use")
	parser.add_argument("--group_name", type=str, default="default", help="Group name for wandb")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False, help="Whether to load checkpoint or not")

	args = parser.parse_args()


	wandb_config = {
		"learningrate": args.learning_rate,
		"freeze": args.freeze,
		"classweights": args.class_weights,
		"model_size": args.model_size,
		"load_checkpoint": args.load_checkpoint
	}

	# wandb_name = "_".join([f"{k}-{v}" for k, v in wandb_config.items()])	
	wandb_name = str(wandb_config["learningrate"])

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	group_name = args.group_name

	if args.logging: 
		wandb.init(
				project="Geospatial_Regression_final",
				group=group_name,
				config = wandb_config, 
				name=wandb_name,
				)
		wandb.run.log_code(".")


	path_train=filter_to_last_month(data_path("training",config["data_dir"]), config["data_dir"], "Northeast")
	path_val=filter_to_last_month(data_path("validation",config["data_dir"]), config["data_dir"], "Northeast")
	path_test=filter_to_last_month(data_path("testing",config["data_dir"]), config["data_dir"], "Northeast")

	cycle_dataset_train=cycle_dataset(path_train,split="train")
	cycle_dataset_val=cycle_dataset(path_val,split="val")
	cycle_dataset_test=cycle_dataset(path_test,split="test")

	train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=2)
	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)

	device = "cuda"
	weights_path = config["pretrained_cfg"]["prithvi_model_new_weight"] if args.load_checkpoint else None
	model=PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=1, model_size=args.model_size) #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
	model=model.to(device)

	model.backbone.model.encoder.eval()
	for blk in model.backbone.model.encoder.blocks:
		for param in blk.parameters():
			param.requires_grad = False
		
	checkpoint_dir = config["training"]["checkpoint_dir"] + f"/Northeast/{group_name}"
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"
	
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

	best_acc_val=0
	for epoch in range(config["training"]["n_iteration"]):

		loss_i=0.0

		total_correct = {i:[] for i in range(4)}

		print("iteration started")
		model.train()

		for j,(input,mask, class_weights) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input=input.to(device)[:, 0]
			mask=mask.to(device)

			optimizer.zero_grad()
			out=model(input)

			loss=segmentation_loss(mask=mask,pred=out,device=device,class_weights=class_weights) if args.class_weights else segmentation_loss(mask=mask,pred=out,device=device,class_weights=None)
			total_correct = compute_accuracy(mask, out, total_correct)
			loss_i += loss.item() * input.size(0)  # Multiply by batch size

			loss.backward()
			optimizer.step()

			if j%10==0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				for idx in range(4): 
					acc_to_print = np.mean(np.concatenate(total_correct[idx]))
					to_print += f"acc_{idx}: {acc_to_print}\n "	
				print(to_print)

		acc_dataset_train = {i:np.mean(np.concatenate(total_correct[i])) for i in range(4)}
		epoch_loss_train = loss_i / len(train_dataloader.dataset)
		epoch_loss_val, acc_dataset_val = eval_data_loader(val_dataloader, model, device)

		if args.logging: 
			to_log = {} 
			to_log["epoch"] = epoch + 1 
			to_log["val_loss"] = epoch_loss_val
			to_log["train_loss"] = epoch_loss_train
			to_log["learning_rate"] = optimizer.param_groups[0]['lr']
			for idx in range(4):
				to_log[f"acc_train_{idx}"] = acc_dataset_train[idx]
				to_log[f"acc_val_{idx}"] = acc_dataset_val[idx]
			wandb.log(to_log)


		print("="*100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"{acc_dataset_train[idx]}, acc_val_{idx}: {acc_dataset_val[idx]} \n "
		print(to_print)
		print("="*100)

		scheduler.step(epoch_loss_val)
		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean>best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint)
			best_acc_val=acc_dataset_val_mean

		if epoch == 1 and (not args.freeze): 
			model.backbone.model.encoder.train()
			for blk in model.backbone.model.encoder.blocks:
				for param in blk.parameters():
					param.requires_grad = True

			print("UnFreezing prithvi model")
			print("="*100)

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	epoch_loss_val, acc_dataset_val = eval_data_loader(val_dataloader, model, device)
	epoch_loss_test, acc_dataset_test = eval_data_loader(test_dataloader, model, device)

	if args.logging:
		for idx in range(4): 
			wandb.run.summary[f"best_acc_val_{idx}"] = acc_dataset_val[idx]
			wandb.run.summary[f"best_acc_test_{idx}"] = acc_dataset_test[idx]
		wandb.run.summary[f"best_avg_acc_val"] = np.mean(list(acc_dataset_val.values()))
		wandb.run.summary[f"best_avg_acc_test"] = np.mean(list(acc_dataset_test.values()))

	if args.logging: 
		wandb.finish()
	
if __name__ == "__main__":
	main()
