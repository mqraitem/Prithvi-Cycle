import os 

# records_dir = "records"
# for freeze in [False]:
#     for model_size in ["300m", "600m"]:
#         for load_checkpoint in [True]:
#             group_name = f"regression_freeze-{freeze}_modelsize-{model_size}_loadcheckpoint-{load_checkpoint}"
#             for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#                 name = f"{group_name}_{learning_rate}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     continue

#                 command = f"qsub -v args=' --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --model_size {model_size} --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/prithvi_cycle_regression.sh"
#                 os.system(command)


records_dir = "records"
for freeze in [False]:
    for model_size in ["300m"]:
        for load_checkpoint in [False, True]:
            group_name = f"regression_freeze-{freeze}_modelsize-{model_size}_loadcheckpoint-{load_checkpoint}"
            for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                name = f"{group_name}_{learning_rate}"
                if os.path.exists(f"{records_dir}/{name}"):
                    continue

                command = f"qsub -v args=' --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --model_size {model_size} --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/prithvi_cycle_regression.sh"
                os.system(command)
