import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

plan_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('conditional', 'cond'),
]

base = {
    'plan': {
        'batch_size': 1,
        'device': 'cuda',  # Change to 'cpu' if no GPU

        ## diffusion model
        'horizon': 96,  # Match training config
        'n_diffusion_steps': 64,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',  # Local logs folder
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'diffusion/H96_T64',  # Match your folder
        'diffusion_epoch': 70000,  # Load 70k checkpoint specifically
    },
}

#------------------------ overrides ------------------------#

maze2d_umaze_v1 = {
    'plan': {
        'horizon': 96,
        'n_diffusion_steps': 64,
    },
}
