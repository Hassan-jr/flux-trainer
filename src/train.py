# import subprocess
# import os
# import yaml

# def fine_tune_function(params, temp_folder_path):
#     """
#     Creates a config.yaml from parameters and runs the training script
#     """
#     # Ensure the temporary folder exists
#     os.makedirs(temp_folder_path, exist_ok=True)
    
#     # Create the config structure matching your original yaml
#     config = {
#         "job": "extension",
#         "config": {
#             "name": params["trigger_word"],
#             "process": [{
#                 "type": "sd_trainer",
#                 "training_folder": "output",
#                 "device": "cuda:0",
#                 "trigger_word": params["trigger_word"],
#                 "network": {
#                     "type": "lora",
#                     "linear": 32,
#                     "linear_alpha": 32,
#                     "dropout": 0.25, 
#                     "network_kwargs": {  
#                         "only_if_contains": [
#                             "transformer.single_transformer_blocks.5.proj_out",
#                             "transformer.single_transformer_blocks.8.proj_out",
#                             "transformer.single_transformer_blocks.9.proj_out",
#                             "transformer.single_transformer_blocks.10.proj_out"
#                         ]
#                     }
#                 },
#                 "save": {
#                     "dtype": "float16",
#                     "save_every": None,
#                     "max_step_saves_to_keep": None,
#                     "push_to_hub": False
#                 },
#                 "datasets": [{
#                     "folder_path": params["datasets"][0]["folder_path"],
#                     "caption_ext": "txt",
#                     "caption_dropout_rate": params["datasets"][0]["caption_dropout_rate"],
#                     "shuffle_tokens": False,
#                     "cache_latents_to_disk": True,
#                     "resolution": [512, 768, 1024]
#                 }],
#                 "train": {
#                     "batch_size": params["train"]["batch_size"],
#                     "steps": params["train"]["steps"],
#                     "gradient_accumulation_steps": 1,
#                     "train_unet": True,
#                     "train_text_encoder": False,
#                     "gradient_checkpointing": True,
#                     "noise_scheduler": "flowmatch",
#                     "optimizer": params["train"]["optimizer"],
#                     "lr": params["train"]["lr"],
#                     "ema_config": {
#                         "use_ema": True,
#                         "ema_decay": 0.99
#                     },
#                     "dtype": "bf16"
#                 },
#                 "model": {
#                     "name_or_path": "black-forest-labs/FLUX.1-schnell",
#                     "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
#                     "is_flux": True,
#                     "quantize": params["model"]["quantize"]
#                 },
#                 "sample": {
#                     "sampler": "flowmatch",
#                     "sample_every": None,
#                     "width": 1024,
#                     "height": 1024,
#                     "prompts": ["a man holding a sign that says, 'this is a sign'"],
#                     "neg": "",
#                     "seed": 42,
#                     "walk_seed": True,
#                     "guidance_scale": 1,
#                     "sample_steps": 4
#                 }
#             }]
#         },
#         "meta": {
#             "name": "[name]",
#             "version": "1.0"
#         }
#     }
    
#     # Save the config file
#     config_path = os.path.join(temp_folder_path, "config.yaml")
#     with open(config_path, 'w') as f:
#         yaml.safe_dump(config, f, sort_keys=False)
    
#     # Get the path to run.py
#     current_path = os.getcwd()
#     run_script_path = os.path.join(current_path, 'ai-toolkit', 'run.py')

#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# def list_files(directory):
#     try:
#         # Get absolute path of the directory
#         abs_directory = os.path.abspath(directory)
#         # List all entries in the directory
#         entries = os.listdir(abs_directory)
#         # Filter out directories, keep only files
#         files = [entry for entry in entries if os.path.isfile(os.path.join(abs_directory, entry))]
#         return files
#     except FileNotFoundError:
#         print(f"Directory not found: {directory}")
#         return []
#     except PermissionError:
#         print(f"Permission denied: {directory}")
#         return []
#     except Exception as e:
#         print(f"An error occurred while listing files in {directory}: {e}")
#         return []

# # Directories to list files from
# directories = [current_path, os.path.join(current_path, 'ai-toolkit')]

# for directory in directories:
#     print(f"Files in '{directory}':")
#     files = list_files(directory)
#     for file in files:
#         print(file)
#     print()

#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
#     # Run the training script
#     cmd = f"python {run_script_path} {config_path}"
    
#     try:
#         result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
#         print(result.stdout)  # Print the output
#         if result.returncode == 0:
#             status = "success"
#         else:
#             status = "failed"
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred: {e}")
#         print(f"Error output: {e.output}")
#         status = "failed"
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         status = "failed"

#     return status

import subprocess
import os
import yaml

def fine_tune_function(params, temp_folder_path):
    """
    Creates a config.yaml from parameters and runs the training script
    """
    # Ensure the temporary folder exists
    os.makedirs(temp_folder_path, exist_ok=True)
    
    # Create the config structure matching your original yaml
    config = {
        "job": "extension",
        "config": {
            "name": params["trigger_word"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": params["training_folder"],
                "device": "cuda:0",
                "trigger_word": params["trigger_word"],
                "network": {
                    "type": "lora",
                    "linear": 256,
                    "linear_alpha": 256,
                    "dropout": 0.25, 
                    "network_kwargs": {  
                        "only_if_contains": [
                            "transformer.single_transformer_blocks.5.proj_out",
                            "transformer.single_transformer_blocks.8.proj_out",
                            "transformer.single_transformer_blocks.9.proj_out",
                            "transformer.single_transformer_blocks.10.proj_out"
                        ]
                    }
                },
                "save": {
                    "dtype": "float32",
                    "save_every": None,
                    "max_step_saves_to_keep": None,
                    "push_to_hub": False
                },
                "datasets": [{
                    "folder_path": params["datasets"][0]["folder_path"],
                    "caption_ext": "txt",
                    "caption_dropout_rate": params["datasets"][0]["caption_dropout_rate"],
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [512, 768, 1024]
                }],
                "train": {
                    "batch_size": params["train"]["batch_size"],
                    "steps": params["train"]["steps"],
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": params["train"]["optimizer"],
                    "lr": params["train"]["lr"],
                    "ema_config": {
                        "use_ema": True,
                        "ema_decay": 0.99
                    },
                    "dtype": "bf16"
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-schnell",
                    "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
                    "is_flux": True,
                    "quantize": params["model"]["quantize"]
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": None,
                    "width": 1024,
                    "height": 1024,
                    "prompts": ["a man holding a sign that says, 'this is a sign'"],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 1,
                    "sample_steps": 4
                }
            }]
        },
        "meta": {
            "name": "[name]",
            "version": "1.0"
        }
    }
    
    # Save the config file
    config_path = os.path.join(temp_folder_path, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    
    # Get the path to run.py
    current_path = os.getcwd()
    run_script_path = os.path.join(current_path, 'ai-toolkit', 'run.py')
    
    # Run the training script
    cmd = f"python {run_script_path} {config_path}"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print the output
        if result.returncode == 0:
            status = "success"
        else:
            status = "failed"
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Error output: {e.output}")
        status = "failed"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        status = "failed"

    return status