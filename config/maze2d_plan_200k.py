import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

logbase = 'logs'

base = {
    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 96,
        'n_diffusion_steps': 64,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'diffusion/H96_T64',
        'diffusion_epoch': 200000,  # Load 200k checkpoint
        
        ## inference settings
        'max_episode_length': 300,
    },
}

#------------------------ overrides ------------------------#

maze2d_umaze_v1 = {
    'plan': {
        'horizon': 96,
        'n_diffusion_steps': 64,
    },
}

maze2d_medium_v1 = {
    'plan': {
        'horizon': 192,
        'n_diffusion_steps': 96,
    },
}

maze2d_large_v1 = {
    'plan': {
        'horizon': 256,
        'n_diffusion_steps': 128,
    },
}
