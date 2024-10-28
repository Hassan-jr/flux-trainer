import torch
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA
from main import lora_train 

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.load_models()

    def load_models(self):
        pass  # Since your generate_image function downloads models, this is not needed

MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #
@torch.inference_mode()
def generate_image_handler(job):
    '''
    Handler for Lora training job
    '''
    job_input = job["input"]
    
    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    job_input = validated_input['validated_input']
    
    # Call your lora_train function with the validated parameters
    result = lora_train(
        name=job_input['name'],
        trigger_word=job_input['trigger_word'],
        model_id=job_input['model_id'],
        caption_dropout_rate=job_input['caption_dropout_rate'],
        batch_size=job_input['batch_size'],
        steps=job_input['steps'],
        optimizer=job_input['optimizer'],
        lr=job_input['lr'],
        quantize=job_input['quantize'],
        r2_bucket_name=job_input['r2_bucket_name'],
        r2_access_key_id=job_input['r2_access_key_id'],
        r2_secret_access_key=job_input['r2_secret_access_key'],
        r2_endpoint_url=job_input['r2_endpoint_url'],
        r2_path_in_bucket=job_input['r2_path_in_bucket']
    )
    
    return result

runpod.serverless.start({"handler": generate_image_handler})