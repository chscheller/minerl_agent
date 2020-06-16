import logging
import logging
import os
from typing import Callable, Tuple

import tensorflow as tf

import aicrowd_helper
from minerl_agent.agent.agent import Agent

logger = logging.getLogger(__name__)


def train(log_dir: str,
          load_dir: str,
          dataset_fn: Callable[[], Tuple[tf.data.Dataset, tuple, tuple, list]],
          agent_fn: Callable[[], Agent],
          learner_fn: Callable[[Agent, tuple], tf.Operation],
          seed: int = 0):
    # tf session config
    filters = []
    config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
    config.gpu_options.allow_growth = True
    server = tf.train.Server.create_local_server(
        config=config
    )

    # build graph and run training
    with tf.Graph().as_default(), tf.device('/cpu'):
        tf.set_random_seed(seed)

        dataset, dataset_shapes, dataset_types, records_meta = dataset_fn()
        dataset_size = sum([record['clipped_episode_length'] for record in records_meta])
        iterator_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(iterator_handle, output_types=dataset_types,
                                                       output_shapes=dataset_shapes)
        train_iterator = dataset.prefetch(1).make_one_shot_iterator()
        learner_inputs = iterator.get_next()

        agent = agent_fn()

        tf.get_variable(
            'num_env_frames',
            initializer=tf.zeros_initializer(),
            shape=[],
            dtype=tf.int64,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        with tf.device('/gpu'):
            train_outputs = learner_fn(agent, learner_inputs)

        scaffold = None

        if load_dir is not None and load_dir != 'None':
            if not os.path.isfile(os.path.join(load_dir, 'checkpoint')):
                raise ValueError(f"No checkpoint found in load_dir '{load_dir}'")

            # variables_to_restore = tf.contrib.framework.get_variables_to_restore()
            init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                tf.train.latest_checkpoint(load_dir),
                tf.trainable_variables(),
                ignore_missing_vars=True
            )
            scaffold = tf.train.Scaffold(
                saver=tf.train.Saver(),
                init_fn=lambda _, sess: sess.run(init_assign_op, init_feed_dict)
            )

        # Create MonitoredSession (to run the graph, checkpoint and log).
        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=True,
                checkpoint_dir=log_dir,
                save_checkpoint_secs=600,
                save_summaries_secs=30,
                log_step_count_steps=10000,
                scaffold=scaffold
        ) as session:
            train_handle = session.run_step_fn(
                lambda step_context: step_context.session.run(train_iterator.string_handle()))
            _ = session.run(train_outputs, {iterator_handle: train_handle})
            epoch = 0
            while not session.should_stop():
                num_frames = session.run(train_outputs, {iterator_handle: train_handle})
                current_epoch = num_frames // dataset_size
                if current_epoch != epoch:
                    logging.info(f"Epoch {num_frames // dataset_size} finished.")
                    epoch = current_epoch
