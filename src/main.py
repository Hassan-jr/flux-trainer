import os
import logging
from train import fine_tune_function
from cloudflare_util import upload
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lora_train(
    name,
    trigger_word,
    model_id,
    caption_dropout_rate=0.0,
    batch_size=1,
    steps=2000,
    optimizer='adamw8bit',
    lr=1e-4,
    quantize=True,
    r2_bucket_name="",
    r2_access_key_id="",
    r2_secret_access_key="",
    r2_endpoint_url="",
    r2_path_in_bucket="Loras"
):
    """
    Lora training function that combines user parameters with default configuration
    """
    

    
    # Folder Paths
    folder_path = os.makedirs(os.path.join(os.getcwd(), "ai-toolkit", "output", model_id), exist_ok=True) or os.path.join(os.getcwd(), "ai-toolkit", "output", model_id)
    temp_folder_path = folder_path #os.path.join(folder_path, model_id) # this is for config
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
            "name_or_path": "black-forest-labs/FLUX.1-schnell",
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
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

    # Start the fine-tuning process
    logging.info('Starting fine-tuning...')
    status = fine_tune_function(fine_tune_params, temp_folder_path)
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
    
    # Clean up temporary folder
    shutil.rmtree(temp_folder_path)
    
    return status