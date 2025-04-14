# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import subprocess
import os
import yaml
import sys # Import sys to get the executable path
import logging # Assuming you have logging configured as in lora_train

# Configure logging if not already done globally
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fine_tune_function(params, temp_folder_path):
    """
    Creates a config.yaml from parameters and runs the training script
    """
    # Ensure the temporary folder exists
    os.makedirs(temp_folder_path, exist_ok=True)

    # --- [Your config dictionary creation remains the same] ---
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
                    "linear": 512,
                    "linear_alpha": 512,
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
                    "token_dropout_rate": 0,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [512, 768, 1024]
                }],
                "train": {
                    "batch_size": params["train"]["batch_size"],
                    "steps": params["train"]["steps"],
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": params["train"]["optimizer"],
                    "lr": params["train"]["lr"],
                    "linear_timesteps": True,
                    "ema_config": {
                        "use_ema": True,
                        "ema_decay": 0.99
                    },
                    "dtype": "bf16"
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",              
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
                    "guidance_scale": 4,
                    "sample_steps": 1
                }
            }]
        },
        "meta": {
            "name": "[name]",
            "version": "1.0"
        }
    }
    # --- [End of config dictionary] ---

    # Save the config file
    config_path_absolute = os.path.join(temp_folder_path, "config.yaml")
    logging.info(f"Saving config to: {config_path_absolute}")
    with open(config_path_absolute, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Determine the ai-toolkit directory path relative to the *current script's location*
    # This might be more robust depending on how your project is structured/deployed
    # current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Use this if fine_tune_function is in a different file
    # toolkit_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'ai-toolkit')) # Adjust relative path if needed

    # Or stick to CWD if lora_train and fine_tune_function are called from the project root ('src')
    original_dir = os.getcwd() # Keep track if needed elsewhere, but avoid chdir for subprocess
    toolkit_dir = os.path.join(original_dir, 'ai-toolkit') # Assumes called from 'src'
    logging.info(f"Using ai-toolkit directory: {toolkit_dir}")

    # Check if toolkit_dir and run.py exist BEFORE trying to run
    run_py_path = os.path.join(toolkit_dir, 'run.py')
    if not os.path.isdir(toolkit_dir):
        logging.error(f"ai-toolkit directory not found at: {toolkit_dir}")
        return "failed"
    if not os.path.isfile(run_py_path):
        logging.error(f"run.py not found at: {run_py_path}")
        # Log contents for debugging
        try:
            logging.info(f"Contents of {toolkit_dir}: {os.listdir(toolkit_dir)}")
        except Exception as list_e:
            logging.error(f"Could not list contents of {toolkit_dir}: {list_e}")
        return "failed"

    # Use the *absolute* path for the config file in the command argument.
    # run.py, when executed with cwd=toolkit_dir, might expect paths relative to toolkit_dir
    # OR it might handle absolute paths correctly. Using absolute is often safer.
    # If run.py *requires* a relative path from toolkit_dir, calculate it:
    config_path_relative = os.path.relpath(config_path_absolute, start=toolkit_dir)
    logging.info(f"Relative config path for command: {config_path_relative}")

    # Build the command as a list
    python_executable = sys.executable # Get absolute path to current python interpreter
    cmd_list = [python_executable, 'run.py', config_path_relative] # Use relative path here
    logging.info(f"Executing command: {' '.join(cmd_list)} in directory: {toolkit_dir}")

    status = "failed" # Default status
    try:
        # Run the command specifying the working directory (cwd)
        result = subprocess.run(
            cmd_list,
            check=True,           # Raise CalledProcessError on non-zero exit
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode stdout/stderr as text
            cwd=toolkit_dir       # Set the working directory for the subprocess
        )
        logging.info("Subprocess stdout:")
        logging.info(result.stdout)
        logging.info("Subprocess stderr:")
        logging.info(result.stderr) # Log stderr even on success, might contain warnings
        status = "success" # Set status to success if check=True passes

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        logging.error(f"Command executed: {' '.join(e.cmd)}")
        logging.error("Subprocess stdout:")
        logging.error(e.stdout)
        logging.error("Subprocess stderr:")
        logging.error(e.stderr) # *** This is crucial for debugging exit code 127 ***
        status = "failed"
    except FileNotFoundError as e:
        # This happens if python_executable or run.py itself wasn't found by subprocess.run
        logging.error(f"FileNotFoundError during subprocess execution: {e}")
        logging.error(f"Attempted to run: {cmd_list}")
        logging.error(f"Working directory: {toolkit_dir}")
        status = "failed"
    except Exception as e:
        logging.error(f"An unexpected error occurred during subprocess execution: {e}")
        status = "failed"
    # No finally block needed for os.chdir

    return status