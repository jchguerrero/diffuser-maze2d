import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    n_runs: int = 1

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')


# Check args
print(f'All args: {vars(args)}')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#------------------------ output setup ------------------------#

## create timestamped run folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
runs_savepath = join(args.logbase, args.dataset, 'plans', f'inference_runs_{timestamp}')
os.makedirs(runs_savepath, exist_ok=True)

## add start/goal markers to images
def add_markers(img_path, start_pos, goal_pos, output_path=None):
    if output_path is None:
        output_path = img_path
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    
    def world_to_pixel(pos):
        normalized_x = (pos[0] + 0.5) / 5.0
        normalized_y = (pos[1] + 0.5) / 5.0
        x_pix = normalized_y * img.shape[1]
        y_pix = normalized_x * img.shape[0]
        
        return x_pix, y_pix
    
    if start_pos is not None:
        x, y = world_to_pixel(start_pos)
        ax.plot(x, y, marker='*', markersize=40, color='lime', 
               markeredgecolor='darkgreen', markeredgewidth=4, zorder=100)
        ax.text(x, y+40, 'START', ha='center', va='top', fontsize=16, 
               color='white', weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='green', 
                        alpha=0.9, edgecolor='darkgreen', linewidth=2))
    
    if goal_pos is not None:
        x, y = world_to_pixel(goal_pos)
        ax.plot(x, y, marker='*', markersize=40, color='red',
               markeredgecolor='darkred', markeredgewidth=4, zorder=100)
        ax.text(x, y-40, 'GOAL', ha='center', va='bottom', fontsize=16,
               color='white', weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red',
                        alpha=0.9, edgecolor='darkred', linewidth=2))
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

## collect results from all runs
all_runs_summary = []

#---------------------------------- main loop ----------------------------------#

## loop for multiple runs
for run_idx in range(args.n_runs):
    
    ## folder for each run
    run_savepath = join(runs_savepath, f'run_{run_idx:03d}')
    os.makedirs(run_savepath, exist_ok=True)
    args.savepath = run_savepath

    observation = env.reset()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]
    
    ## trajectory data for CSV
    trajectory_data = []
    
    ## start and goal positions
    start_pos = observation[:2].copy()
    goal_pos = env._target

    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation

            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        
        ## log trajectory data
        distance_to_goal = np.linalg.norm(observation[:2] - np.array(goal_pos))
        trajectory_data.append({
            'timestep': t,
            'pos_x': float(observation[0]),
            'pos_y': float(observation[1]),
            'vel_x': float(observation[2]) if len(observation) > 2 else 0.0,
            'vel_y': float(observation[3]) if len(observation) > 2 else 0.0,
            'action_x': float(action[0]),
            'action_y': float(action[1]),
            'reward': float(reward),
            'cumulative_reward': float(total_reward + reward),
            'distance_to_goal': float(distance_to_goal),
            'terminal': bool(terminal),
        })
        
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action}'
        )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')

            if t == 0: 
                renderer.composite(fullpath, samples.observations, ncol=1)
                ## save plan with markers
                fullpath_marked = join(args.savepath, 'plan_marked.png')
                add_markers(fullpath, start_pos, goal_pos, fullpath_marked)


            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
            ## save rollout with markers
            rollout_marked_path = join(args.savepath, 'rollout_marked.png')
            add_markers(join(args.savepath, 'rollout.png'), start_pos, goal_pos, rollout_marked_path)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    
    ## save trajectory CSV
    csv_path = join(args.savepath, 'trajectory.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        if len(trajectory_data) > 0:
            fieldnames = trajectory_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory_data)
    
    ## save summary JSON
    final_distance = float(np.linalg.norm(observation[:2] - np.array(goal_pos)))
    success = bool(terminal and reward > 0 and final_distance < 0.5)
    summary = {
        'run_index': run_idx,
        'start_pos': [float(x) for x in start_pos],
        'goal_pos': [float(x) for x in goal_pos],
        'initial_distance': float(np.linalg.norm(start_pos - np.array(goal_pos))),
        'final_distance': final_distance,
        'total_reward': float(total_reward),
        'score': float(score),
        'timesteps': int(t + 1),
        'terminal': bool(terminal),
        'success': success,
    }
    summary_path = join(args.savepath, 'summary.json')
    json.dump(summary, open(summary_path, 'w'), indent=2)
    all_runs_summary.append(summary)

#------------------------ results ------------------------#

results = {
    'n_runs': args.n_runs,
    'dataset': args.dataset,
    'diffusion_epoch': diffusion_experiment.epoch,
    'timestamp': timestamp,
    'runs': all_runs_summary,
    'statistics': {
        'success_rate': sum(1 for r in all_runs_summary if r['success']) / args.n_runs,
        'avg_score': float(np.mean([r['score'] for r in all_runs_summary])),
        'std_score': float(np.std([r['score'] for r in all_runs_summary])),
        'avg_steps': float(np.mean([r['timesteps'] for r in all_runs_summary])),
        'avg_final_distance': float(np.mean([r['final_distance'] for r in all_runs_summary])),
    }
}
results_path = join(runs_savepath, 'results_summary.json')
json.dump(results, open(results_path, 'w'), indent=2)

print(f'\nAll {args.n_runs} runs complete')
print(f'Success rate: {results["statistics"]["success_rate"]*100:.1f}%')
print(f'Average score: {results["statistics"]["avg_score"]:.4f}')
print(f'Results: {runs_savepath}\n')