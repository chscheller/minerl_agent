import functools
import logging
import os
import pathlib
from multiprocessing import freeze_support
from typing import Callable

import coloredlogs
import minerl
import sonnet as snt
import tensorflow as tf
from absl import flags

from minerl_agent.behaviour_cloning import behaviour_cloning, dataset, learner
from minerl_agent.environment.actions import ActionSpace
from minerl_agent.environment.observations import ObservationSpace
from minerl_agent.impala import impala
from utility.parser import Parser

FLAGS = flags.FLAGS

# behaviour cloning
flags.DEFINE_bool('bc', default=True, help='behaviour cloning')
flags.DEFINE_multi_string('bc_environments',
                          default=['MineRLObtainDiamond-v0', 'MineRLObtainIronPickaxe-v0', 'MineRLTreechop-v0'],
                          # default=['MineRLTreechop-v0'],
                          help='bc environments')
flags.DEFINE_integer('bc_min_score', default=1, help='minimum score for demonstrations to be considered')
flags.DEFINE_integer('bc_max_sequence_length', default=1000, help='sequence clipping length for demonstrations')
flags.DEFINE_integer('bc_num_pipeline_workers', default=10, help='number of parallel workers for the preprocessing')
flags.DEFINE_integer('bc_num_epochs', default=125, help='number of bc epochs')
flags.DEFINE_integer('bc_batch_size', default=3, help='batch size')  # 15
flags.DEFINE_float('bc_learning_rate', default=0.0005, help='Learning rate')
flags.DEFINE_bool('bc_ignore_unimportant_actions', default=True, help='ignore unimportant actions during subsampling')

# impala
flags.DEFINE_bool('impala', default=True, help='impala')
flags.DEFINE_integer('impala_num_actors', default=5, help='number of impala actors')
flags.DEFINE_integer('impala_batch_size', default=16, help='batch size')  # 64
flags.DEFINE_integer('impala_unroll_length', default=50, help='unroll length')
flags.DEFINE_enum('impala_reward_clipping', default='default', enum_values=['abs_one', 'soft_asymmetric', 'default'],
                  help='reward clipping (default for no clipping)')
flags.DEFINE_float('impala_discounting', default=0.99, help='discounting')
flags.DEFINE_integer('impala_total_environment_frames', default=int(7.5e6), help='total environment frames')
flags.DEFINE_float('impala_baseline_cost', default=0.5, help='baseline cost')
flags.DEFINE_float('impala_entropy_cost', default=2e-05, help='entropy cost')
flags.DEFINE_float('impala_policy_cloning_cost', default=0.01, help='CLEAR policy cloning cost')
flags.DEFINE_float('impala_value_cloning_cost', default=0.005, help='CLEAR value cloning cost')
flags.DEFINE_float('impala_clip_grad_norm', default=100., help='gradient clipping norm')
flags.DEFINE_bool('impala_clip_advantage', default=True, help='advantage clipping')
flags.DEFINE_float('impala_learning_rate', default=0.0001, help='impala learning rate')
flags.DEFINE_integer('impala_replay_buffer_size', default=6000, help='experience buffer size (number of sequences)')
flags.DEFINE_integer('impala_max_episode_length', default=-1, help='maximum number of frames per episode')
flags.DEFINE_string('impala_environment_name', default='MineRLObtainDiamond-v0', help='Training environment')
flags.DEFINE_integer('impala_num_critic_pretrain_frames', default=int(5e5), help='num_critic_pretrain_frames')
flags.DEFINE_float('impala_replay_proportion', default=0.9375, help='replay_proportion')
flags.DEFINE_float('impala_reward_scaling', default=1.0, help='reward_scaling')

coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
# MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/christian/source/MineRL/data')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a
# tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
                initial_poll_timeout=600)


def main(log_dir, load_dir, observation_space: ObservationSpace, action_space: ActionSpace, max_step_mul: int,
         fixed_step_mul: bool, step_mul: int, agent_fn: Callable[[], snt.RNNCore], seed=0, malmo_base_port: int = None):
    """
    This function will be called for training phase.
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    data_path = log_dir
    bc_log_dir = os.path.join(log_dir, 'bc')
    impala_log_dir = os.path.join(log_dir, 'impala')

    if FLAGS.bc:
        pathlib.Path(bc_log_dir).mkdir(parents=True, exist_ok=True)

        # generate tfrecords dataset from minerl data pipelines:
        data_pipelines = [
            minerl.data.make(env, data_dir=MINERL_DATA_ROOT, num_workers=FLAGS.bc_num_pipeline_workers)
            for env in FLAGS.bc_environments
        ]
        dataset_path = dataset.create_tfrecords_form_minerl_pipelines(
            data_pipelines=data_pipelines, data_path=data_path, name='train', observation_space=observation_space,
            action_space=action_space, max_step_mul=max_step_mul, max_sequence_length=FLAGS.bc_max_sequence_length,
            ignore_unimportant_actions=FLAGS.bc_ignore_unimportant_actions, seed=seed)

        # run behaviour cloning training on tfrecords dataset:
        dataset_fn = functools.partial(
            dataset.make_from_tfrecors,
            dataset_path=dataset_path, name='train', observation_space=observation_space, action_space=action_space,
            min_score=FLAGS.bc_min_score, batch_size=FLAGS.bc_batch_size, num_epochs=FLAGS.bc_num_epochs)
        learner_fn = functools.partial(
            learner.build_learner,
            learning_rate=FLAGS.bc_learning_rate, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
            clip_grad_norm=100)
        behaviour_cloning.train(
            log_dir=bc_log_dir, load_dir=load_dir, dataset_fn=dataset_fn, agent_fn=agent_fn, learner_fn=learner_fn,
            seed=seed)

        load_dir = bc_log_dir

    if FLAGS.impala:
        pathlib.Path(impala_log_dir).mkdir(parents=True, exist_ok=True)

        impala.train(impala_log_dir, load_dir, FLAGS.impala_environment_name, action_space, observation_space,
                     agent_fn=agent_fn, num_actors=FLAGS.impala_num_actors, batch_size=FLAGS.impala_batch_size,
                     unroll_length=FLAGS.impala_unroll_length, reward_clipping=FLAGS.impala_reward_clipping,
                     discounting=FLAGS.impala_discounting,
                     total_environment_frames=FLAGS.impala_total_environment_frames,
                     baseline_cost=FLAGS.impala_baseline_cost, entropy_cost=FLAGS.impala_entropy_cost,
                     policy_cloning_cost=FLAGS.impala_policy_cloning_cost,
                     value_cloning_cost=FLAGS.impala_value_cloning_cost,
                     clip_grad_norm=FLAGS.impala_clip_grad_norm, clip_advantage=FLAGS.impala_clip_advantage,
                     learning_rate=FLAGS.impala_learning_rate, replay_buffer_size=FLAGS.impala_replay_buffer_size,
                     fixed_step_mul=fixed_step_mul, step_mul=step_mul,
                     num_critic_pretrain_frames=FLAGS.impala_num_critic_pretrain_frames,
                     max_episode_length=FLAGS.impala_max_episode_length,
                     replay_proportion=FLAGS.impala_replay_proportion,
                     reward_scaling=FLAGS.impala_reward_scaling,
                     malmo_base_port=malmo_base_port)


if __name__ == '__main__':
    freeze_support()
