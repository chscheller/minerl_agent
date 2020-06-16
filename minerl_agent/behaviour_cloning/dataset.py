import functools
import json
import logging
import multiprocessing
import os
import pathlib
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from minerl.data import DataPipeline
from minerl.env import spaces

from minerl_agent.behaviour_cloning.tfrecrods import serialize_sequence_example, parse_sequence_example
from minerl_agent.environment.actions import skip_action_repeats_accumulate_camera, ActionSpace
from minerl_agent.environment.observations import ObservationSpace

logger = logging.getLogger(__name__)


def _preprocess_function(out_path, name, observation_space: ObservationSpace, action_space: ActionSpace, max_step_mul,
                         max_sequence_length, ignore_unimportant_actions, inputs):
    index, sequence = inputs
    states, actions, rewards, _, dones, meta = sequence  # drop s' from (s, a, r, s', d, m)

    # skip aciton repeats and accumulate camera actions
    states, actions, rewards, done, step_muls = skip_action_repeats_accumulate_camera(
        (states, actions, rewards, dones), max_step_mul, action_space.camera_max_angle(),
        action_space.unimportant_actions(), ignore_unimportant_actions)
    sequence_length = len(rewards)

    # apply observation and action space transformations
    states = observation_space.transform_to(states)
    actions = action_space.transform(actions)

    # truncate sequences to max_sequence_length
    sequence = states, actions, rewards, dones, step_muls
    sequence = tf.nest.map_structure(lambda x: x[:max_sequence_length], sequence)

    # create tfrecords sample
    sequence_example = serialize_sequence_example(sequence)
    path = os.path.join(out_path, f'{name}.{index}.tfrecords')
    with tf.python_io.TFRecordWriter(path) as writer:
        writer.write(sequence_example.SerializeToString())

    score, capped_sequence_length = float(sum(sequence[2])), len(sequence[2])
    return path, score, sequence_length, meta, step_muls, capped_sequence_length


def create_tfrecords_form_minerl_pipelines(data_pipelines: List[DataPipeline], data_path: str, name: str,
                                           observation_space: ObservationSpace, action_space: ActionSpace, max_step_mul,
                                           max_sequence_length: int, ignore_unimportant_actions: bool, seed: int = 0):
    """ Convert MineRL demonstrations to tfrecords.
    Args:
        data_pipelines (List[DataPipeline]): minerl dataset pipeline.
        data_path (str): path where the tfrecords are written to.
        name (str): dataset name
        observation_space (ObservationSpace): target observation space
        action_space (ActionSpace): target action space
        max_step_mul (int): maximum number of steps an action can be repeated.
        max_sequence_length (int): maximum sequence length (sequences that exceed this limit are truncated)
        ignore_unimportant_actions (bool): if true, unimportant actions (according to the action space) are skipped.
        seed (int): random seed

    Returns: A dictionary containing meta data of the generated tfrecords files
    """
    dataset_path = os.path.join(data_path, 'sequential_dataset')
    meta_file_path = os.path.join(dataset_path, f'{name}.meta')

    if os.path.exists(dataset_path):
        logger.info(f'Dataset at {dataset_path} already exists.')
        return dataset_path

    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
    records, scores, episode_lengths = [], [], []
    step_mul_history = []
    with multiprocessing.Pool(processes=1) as p:
        worker = functools.partial(_preprocess_function, dataset_path, name, observation_space, action_space,
                                   max_step_mul, max_sequence_length, ignore_unimportant_actions)
        for data in data_pipelines:
            inputs = enumerate(
                data.sarsd_iter(num_epochs=1, max_sequence_len=-1, queue_size=1, seed=seed, include_metadata=True),
                start=len(records))
            for path, score, episode_length, meta, step_muls, clipped_episode_length in p.imap(worker, inputs):
                step_mul_history.append(step_muls)
                scores.append(score)
                episode_lengths.append(episode_length)
                records.append(
                    {'path': path, 'score': score, 'episode_length': episode_length,
                     'clipped_episode_length': clipped_episode_length, 'minerl_meta': meta})
    meta = {
        'records': records,
        'max_step_mul': max_step_mul,
        'max_sequence_length': max_sequence_length,
        'scores': {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'max': float(np.max(scores)),
            'min': float(np.min(scores)),
            'std': float(np.std(scores)),
            'episode_scores': scores,
        },
        'episode_lengths': {
            'mean': float(np.mean(episode_lengths)),
            'median': float(np.median(episode_lengths)),
            'max': float(np.max(episode_lengths)),
            'min': float(np.min(episode_lengths)),
            'std': float(np.std(episode_lengths)),
            'episode_episode_lengths': episode_lengths,
        }
    }

    with open(meta_file_path, mode='w') as f:
        json.dump(meta, f, indent=4)

    return dataset_path


def shapes(observation_space: ObservationSpace, action_space: ActionSpace, batch_size=None):
    batch_size = [] if batch_size is None else [batch_size]

    def _tensor_shape(space: spaces.Dict):
        if isinstance(space, spaces.Dict):
            return {k: _tensor_shape(v) for k, v in space.spaces.items()}
        elif space.shape:
            return tf.TensorShape(batch_size + [None] + list(space.shape))
        else:
            return tf.TensorShape(batch_size + [None])

    context_features = {
        'sequence_length': tf.TensorShape(batch_size + []),
        'score': tf.TensorShape(batch_size + []),
    }
    sequence_features = {
        'observation': tf.nest.map_structure(_tensor_shape, observation_space.specs()),
        'action': tf.nest.map_structure(_tensor_shape, action_space.specs()),
        "reward": tf.TensorShape(batch_size + [None]),
        "done": tf.TensorShape(batch_size + [None]),
        "step_mul": tf.TensorShape(batch_size + [None]),
    }
    return context_features, sequence_features


def types(observation_space: ObservationSpace, action_space: ActionSpace):
    def _tensor_type(space: spaces.Dict):
        if isinstance(space, spaces.Dict):
            return {k: _tensor_type(v) for k, v in space.spaces.items()}
        elif space.dtype and space.dtype != np.int64:
            return tf.dtypes.as_dtype(space.dtype)
        else:
            return tf.int32

    context_features = {
        'sequence_length': tf.int32,
        'score': tf.float32,
    }
    sequence_features = {
        'observation': tf.nest.map_structure(_tensor_type, observation_space.specs()),
        'action': tf.nest.map_structure(_tensor_type, action_space.specs()),
        "reward": tf.float32,
        "done": tf.bool,
        "step_mul": tf.int32,
    }
    return context_features, sequence_features


def padded_batch(dataset, batch_size, types_in, shapes_in):
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=shapes_in,
                                   padding_values=tf.nest.map_structure(lambda x: tf.cast(0, dtype=x), types_in),
                                   drop_remainder=True)

    shapes_out = tf.nest.map_structure(lambda ts: tf.TensorShape([batch_size]).concatenate(ts), shapes_in)

    return dataset, shapes_out, types_in


def make_from_tfrecors(dataset_path: str, name: str, observation_space: ObservationSpace, action_space: ActionSpace,
                       min_score: int, batch_size, num_epochs: int) -> Tuple[tf.data.Dataset, tuple, tuple, list]:
    with open(os.path.join(dataset_path, f'{name}.meta'), mode='r') as f:
        meta = json.load(f)

    logger.info(f"Loaded {len(meta['records'])} episodes.")

    records = [record for record in meta['records'] if record['score'] >= min_score]

    logger.info(f"{len(records)}/{len(meta['records'])} episodes remain after filtering. " +
                f"(min_score = {min_score})")

    tfrecords = [record['path'] for record in records]

    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=50, reshuffle_each_iteration=True)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(functools.partial(parse_sequence_example, observation_space.specs(), action_space.specs()))
    dataset, shapes_out, types_out = padded_batch(dataset,
                                                  batch_size=batch_size,
                                                  types_in=types(observation_space, action_space),
                                                  shapes_in=shapes(observation_space, action_space))
    return dataset, shapes_out, types_out, records
