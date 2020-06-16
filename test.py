# Simple env test.
import datetime
import json
import logging
import os
from collections import defaultdict
from multiprocessing import freeze_support

import coloredlogs
import gym
import numpy as np
import scipy.stats
import tensorflow as tf
from minerl.env import spaces

from minerl_agent.agent.agent import AgentOutput
from minerl_agent.environment.actions import ActionSpace
from minerl_agent.environment.observations import ObservationSpace
from utility.utils import gym_space_to_dict

coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 100))


def main(log_dir: str, test_model: str, observation_space: ObservationSpace, action_space: ActionSpace,
         fixed_step_mul: bool, step_mul: int, agent_fn):
    """
    This function will be called for training phase.
    """
    model_dir = os.path.join(log_dir, test_model)

    if not os.path.isfile(os.path.join(model_dir, 'checkpoint')):
        raise ValueError(f"No checkpoint found in model_dir '{model_dir}'")

    # build graph and run training
    with tf.Graph().as_default(), tf.device('/gpu'):

        agent = agent_fn()

        prev_actions = tf.nest.map_structure(
            lambda x: tf.placeholder(dtype=tf.dtypes.as_dtype(x.dtype), shape=[1]),
            gym_space_to_dict(action_space.specs())
        )

        env_outputs = (
            tf.placeholder(dtype=tf.float32, shape=[1]),
            tf.placeholder(dtype=tf.bool, shape=[1]),
            tf.nest.map_structure(lambda x: tf.placeholder(
                dtype=tf.dtypes.as_dtype(x.dtype),
                shape=[1] if x.dtype == spaces.Enum else [1] + list(s for s in x.shape if len(x.shape) >= 3 or s > 1)
            ), gym_space_to_dict(observation_space.specs())),
        )

        initial_agent_state = agent.initial_state(1)
        # env_outputs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=-1) if len(x.get_shape()) < 3 else x,
        #                                  env_outputs)

        def create_state(t):
            with tf.variable_scope(None, default_name='state'):
                return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

        persistent_state = tf.nest.map_structure(create_state, initial_agent_state)
        agent_state = tf.nest.map_structure(lambda v: v.read_value(), persistent_state)
        outputs, agent_state = agent((prev_actions, env_outputs), agent_state)  # type: AgentOutput, any
        assign_state_ops = tf.nest.map_structure(lambda v, t: v.assign(t), persistent_state, agent_state)

        # tf session config
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        total_rewards = []

        reward_time_stamps = defaultdict(list)

        with tf.train.SingularMonitoredSession(
                checkpoint_dir=model_dir,
                config=config
        ) as session:
            flat_placeholders = tf.nest.flatten((prev_actions, env_outputs))
            env = gym.make(MINERL_GYM_ENV)
            for i in range(MINERL_MAX_EVALUATION_EPISODES):
                obs = env.reset()
                action = tf.nest.map_structure(lambda a: a.no_op(), action_space.specs())
                done, reward, total_reward, step = False, 0, 0, 0
                while not done:
                    obs['equipped_items']['mainhand'] = {
                        'damage': obs['equipped_items']['mainhand'].get('damage', 0),
                        'maxDamage': obs['equipped_items']['mainhand'].get('maxDamage', 0),
                        'type': obs['equipped_items']['mainhand']['type']
                        if isinstance(obs['equipped_items']['mainhand']['type'], int)
                        else observation_space.specs()['equipped_items']['mainhand']['type'].n - 1
                    }
                    obs = observation_space.transform_to(obs)
                    flat_inputs = tf.nest.flatten((action, (reward, step == 0, obs)))
                    action, step_mul_, _ = session.run((outputs.action, outputs.step_mul, assign_state_ops),
                                                      feed_dict={p: [v] if len(p.get_shape()) > 2 else [v] for p, v in
                                                                 zip(flat_placeholders, flat_inputs)})
                    if fixed_step_mul:
                        # if step_mul is fixed, step_mul from the actor is overridden by the fixed step_mul
                        step_mul_ = step_mul - 1
                    action = tf.nest.map_structure(lambda a: int(np.squeeze(a)), action)
                    step_mul_ = int(np.squeeze(step_mul_))
                    gym_action = action_space.transform_back(action)
                    gym_action['camera'] = np.asarray(gym_action['camera']) / (step_mul_ + 1)
                    for j in range(step_mul_ + 1):
                        if done:
                            break
                        obs, reward, done, info = env.step(gym_action)
                        if j == 0 and fixed_step_mul:
                            # if step_mul is fixed, we need to reset "one_step" actions after the first step:
                            for action_name in ["jump", "place", "craft", "nearbyCraft", "nearbySmelt"]:
                                gym_action[action_name] = 0
                        total_reward += reward
                        if reward > 0:
                            logging.info(f'Reward: {reward} (Step: {step}, Total reward: {total_reward})')
                            reward_time_stamps[str(int(reward))].append(step)
                        step += 1
                total_rewards.append(total_reward)
                logging.info(f"Episode {i + 1} of {MINERL_MAX_EVALUATION_EPISODES} finished. Score: {total_reward} "
                             f"(Mean: {np.mean(total_rewards)})")

        logging.info(f"Mean score: {np.mean(total_rewards)}")

        out_path = os.path.join(model_dir, f'test_outcome_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        with open(out_path, mode='w') as f:
            json.dump({
                'num_episodes': len(total_rewards),
                'epsiode_return_mean': float(np.mean(total_rewards)),
                'epsiode_return_median': float(np.median(total_rewards)),
                'epsiode_return_std': float(np.std(total_rewards)),
                'epsiode_return_min': float(min(total_rewards)),
                'epsiode_return_max': float(max(total_rewards)),
                'epsiode_return_mean_ci95': [float(x) for x in scipy.stats.t.interval(
                    0.95, len(total_rewards) - 1, loc=np.mean(total_rewards), scale=scipy.stats.sem(total_rewards)
                )],
                'epsidoe_returns': [float(r) for r in total_rewards],
                'reward_time_stamps': reward_time_stamps
            }, f, indent=4)

    env.close()


if __name__ == '__main__':
    freeze_support()
    main()
