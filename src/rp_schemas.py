INPUT_SCHEMA = {
    'image_urls': {
        'type': list,
        'required': True,
        'default': [] # images to train
    },
    'trigger_word': {
        'type': str,
        'required': True,
        # 'description': 'Trigger word to be added to captions'
    },
    'model_id': {
        'type': str,
        'required': True,
        # 'description': 'model id for database'
    },
    'caption_dropout_rate': {
        'type': float,
        'required': False,
        'default': 0.0,
        # 'constraints': lambda x: 0 <= x <= 1,
        # 'description': 'Rate at which captions will be dropped during training'
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 1,
        # 'constraints': lambda x: x > 0,
        # 'description': 'Batch size for training'
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 2000,
        # 'constraints': lambda x: 500 <= x <= 4000,
        # 'description': 'Total number of training steps'
    },
    'optimizer': {
        'type': str,
        'required': False,
        'default': 'adamw8bit',
        # 'constraints': lambda x: x in ['adamw8bit', 'adamw', 'lion', 'adamw32bit', 'prodigy'],
        # 'description': 'Optimizer to use for training'
    },
    'lr': {
        'type': float,
        'required': False,
        'default': 1e-4,
        # 'constraints': lambda x: 0 < x < 1,
        # 'description': 'Learning rate for training'
    },
    'quantize': {
        'type': bool,
        'required': False,
        'default': False,
        # 'description': 'Whether to run 8bit mixed precision'
    },
    'r2_bucket_name': {
        'type': str,
        'required': True
    },
    'r2_access_key_id': {
        'type': str,
        'required': True
    },
    'r2_secret_access_key': {
        'type': str,
        'required': True
    },
    'r2_endpoint_url': {
        'type': str,
        'required': True
    },
    'r2_path_in_bucket': {
        'type': str,
        'required': False,
        'default': "Loras"
    },
}