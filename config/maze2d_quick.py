import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('conditional', 'cond'),
]

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training - QUICK TEST SETTINGS
        'n_steps_per_epoch': 100,  # Was: 10000 → faster epochs
        'loss_type': 'l2',
        'n_train_steps': 1000,  # Was: 2e6 → QUICK TEST!
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 100,  # Was: 1000 → save more often
        'sample_freq': 100,  # Was: 1000 → sample more often
        'n_saves': 10,  # Was: 50 → fewer checkpoints
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
        'log_freq': 10,  # Log every 10 steps
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',
        'conditional': False,
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },
}

#------------------------ overrides ------------------------#

'''
    Quick test configs for maze2d environments
    
    Usage:
        python scripts/train.py --config config.maze2d_quick --dataset maze2d-umaze-v1
        python scripts/train.py --config config.maze2d_quick --dataset maze2d-medium-v1
        python scripts/train.py --config config.maze2d_quick --dataset maze2d-large-v1
    
    Training time: ~5-10 minutes (vs 6-8 hours with full config)
    
    For full training, use: config.maze2d
'''

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

# Quick test for umaze
maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        # Quick test settings inherited from base
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

# Quick test for medium
maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 256,
        'n_diffusion_steps': 128,
        # Quick test settings inherited from base
    },
    'plan': {
        'horizon': 256,
        'n_diffusion_steps': 128,
    },
}

# Quick test for large
maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
        # Quick test settings inherited from base
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
