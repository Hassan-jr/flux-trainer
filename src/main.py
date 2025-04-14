import os
import logging
from train import fine_tune_function
from cloudflare_util import upload
from dataset import download_images
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_files_in_dir(directory):
    try:
        all_entries = os.listdir(directory)
        files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory, entry))]
        logging.info(f"Files in '{directory}':")
        for file in files:
            logging.info(f" - {file}")
    except Exception as e:
        logging.info(f"Error listing files in '{directory}': {e}")


def lora_train(
    image_urls,
    trigger_word,
    model_id,
    caption_dropout_rate,
    batch_size,
    steps,
    optimizer,
    lr,
    quantize,
    r2_bucket_name,
    r2_access_key_id,
    r2_secret_access_key,
    r2_endpoint_url,
    r2_path_in_bucket
):
    """
    Lora training function that combines user parameters with default configuration
    """
    

    
    # Folder Paths
    folder_path = os.makedirs(os.path.join(os.getcwd(), "ai-toolkit", "output", model_id), exist_ok=True) or os.path.join(os.getcwd(), "ai-toolkit", "output", model_id)
    dataset_folder_path = os.path.join(folder_path, "images") 
    model_folder_path = os.path.join(folder_path, trigger_word)
    
    lora_path = os.path.join(model_folder_path, f"{trigger_word}.safetensors") 
    
    fine_tune_params = {
        "type": "sd_trainer",
        "training_folder": folder_path,
        "device": "cuda:0",
        "trigger_word": trigger_word,
        
        # Network configuration
        "network": {
            "type": "lora",
            "linear": 16,
            "linear_alpha": 16
        },
        
        # Save configuration
        "save": {
            "dtype": "float16",
            "save_every": 250,
            "max_step_saves_to_keep": 4,
            "push_to_hub": False
        },
        
        # Dataset configuration
        "datasets": [{
            "folder_path": dataset_folder_path,
            "caption_ext": "txt",
            "caption_dropout_rate": caption_dropout_rate,
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": [512, 768, 1024]
        }],
        
        # Training configuration
        "train": {
            "batch_size": batch_size,
            "steps": steps,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "optimizer": optimizer,
            "lr": lr,
            "ema_config": {
                "use_ema": True,
                "ema_decay": 0.99
            },
            "dtype": "bf16"
        },
        
        # Model configuration
        "model": {
            "name_or_path": "black-forest-labs/FLUX.1-dev",
            "is_flux": True,
            "quantize": quantize
        },
        
        # Sample configuration
        "sample": {
            "sampler": "flowmatch",
            "sample_every": 250,
            "width": 1024,
            "height": 1024,
            "prompts": ["a man holding a sign that says, 'this is a sign'"],
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": 1,
            "sample_steps": 4
        }
    }

     # Call the function with the images, folder path, and trigger word
    logging.info('Preparing Dataset...')
    download_images(image_urls, dataset_folder_path, trigger_word)
    logging.info('Dataset Preparation Done...')


    # Start the fine-tuning process
    logging.info('Starting fine-tuning...')
    status = fine_tune_function(fine_tune_params, folder_path)
    logging.info(f"Training status: {status}")

    # Check the status of training
    if status == "success":
        logging.info("Training completed successfully")
        # Upload when training is successful
        if all([r2_bucket_name, r2_access_key_id, r2_secret_access_key, r2_endpoint_url]):
            result = upload(
                bucket_name=r2_bucket_name,
                access_key_id=r2_access_key_id,
                secret_access_key=r2_secret_access_key,
                endpoint_url=r2_endpoint_url,
                file_path=lora_path,
                r2_path_in_bucket=r2_path_in_bucket,
                unique_id=model_id,
                async_upload=False
            )
            
            if result is not None:
                success, message = result
                if success:
                    logging.info("Train and Upload was successful!")
                else:
                    logging.error(f"Upload failed When Training Was Success: {message}")
            else:
                logging.info("Async upload initiated. Check logs for results.")
    else:
        logging.error("Training failed, Nothing Uploaded!")
        # Log files in current working directory (assumed to be 'src')
        log_files_in_dir(os.getcwd())
        # Log files in ai-toolkit folder
        ai_toolkit_dir = os.path.join(os.getcwd(), 'ai-toolkit')
        if os.path.isdir(ai_toolkit_dir):
            log_files_in_dir(ai_toolkit_dir)
        else:
            print(f"'ai-toolkit' directory not found at {ai_toolkit_dir}")
    
    # Clean up temporary folder
    shutil.rmtree(folder_path)
    
    return status