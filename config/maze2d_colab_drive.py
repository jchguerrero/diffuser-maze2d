import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

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
        'logbase': '/content/drive/MyDrive/Colab Notebooks/DiffuserMaze/logs',  # Save to Drive!
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training - OPTIMIZED FOR COLAB FREE (4-6 hours)
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 200000,  # 200k instead of 2M (10x faster)
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,  # Save checkpoints every 10k (20 total)
        'sample_freq': 2000,  # Generate visualizations every 2k (100 images) 🎨
        'n_saves': 20,  # Keep 20 checkpoints
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': '/content/drive/MyDrive/Colab Notebooks/DiffuserMaze/logs',  # Save to Drive!
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'diffusion/H128_T64',
        'diffusion_epoch': 'latest',
    },
}

#------------------------ overrides ------------------------#

'''
    OPTIMIZED FOR COLAB FREE TIER
    
    All configs designed to finish within ~6-10 hours on T4 GPU
    Trade-off: Slightly lower performance vs much faster training
'''

# UMAZE - Simplest maze
maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 96,  # Reduced from 128 (episodes ~150 steps)
        'n_diffusion_steps': 64,
        'dim_mults': (1, 2, 4),  # Smaller model
        'n_train_steps': 200000,  # 200k steps, ~4-6 hours
    },
    'plan': {
        'horizon': 96,
        'n_diffusion_steps': 64,
    },
}

# MEDIUM - Medium complexity maze
maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 192,  # Reduced from 256 (episodes ~250 steps)
        'n_diffusion_steps': 96,  # Reduced from 128
        'dim_mults': (1, 2, 4, 8),  # Keep reasonable size
        'n_train_steps': 400000,  # 400k steps, ~6-8 hours
        'save_freq': 20000,  # Save checkpoints every 20k
        'sample_freq': 4000,  # Visualizations every 4k (100 images) 🎨
    },
    'plan': {
        'horizon': 192,
        'n_diffusion_steps': 96,
    },
}

# LARGE - Most complex maze
maze2d_large_v1 = {
    'diffusion': {
        'horizon': 256,  # Reduced from 384 (episodes ~600 steps)
        'n_diffusion_steps': 128,  # Reduced from 256
        'dim_mults': (1, 2, 4, 8),  # Keep model size reasonable
        'n_train_steps': 500000,  # 500k steps, ~8-10 hours
        'save_freq': 25000,  # Save checkpoints every 25k
        'sample_freq': 5000,  # Visualizations every 5k (100 images) 🎨
        'batch_size': 24,  # Slightly smaller batch (memory)
    },
    'plan': {
        'horizon': 256,
        'n_diffusion_steps': 128,
    },
}

'''
EXPECTED PERFORMANCE vs TRAINING TIME:

Environment | Steps  | Time (T4) | Expected Score | Original Score
------------|--------|-----------|----------------|---------------
umaze       | 200k   | 4-6 hrs   | 70-85%         | 85-95%
medium      | 400k   | 6-8 hrs   | 65-80%         | 80-90%
large       | 500k   | 8-10 hrs  | 60-75%         | 75-85%

NOTES:
- Original configs need 2M steps (30+ hours for large!)
- These optimized configs are 4-10x faster
- Performance drop is acceptable for learning/testing
- For production, use full configs on paid GPU instances
'''
