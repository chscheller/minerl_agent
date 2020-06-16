import logging

import numpy as np
import tensorflow as tf
from minerl.env import spaces

from utility.utils import flatten_nested_dicts, unflatten_nested_dicts

logger = logging.getLogger(__name__)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _float_feature_list(values):
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def serialize_sequence_example(sequence) -> tf.train.SequenceExample:
    def _create_feature(values: np.ndarray):
        if len(values.shape) > 1:
            return _bytes_feature_list([value.tobytes() for value in values])
        elif values.dtype == np.float32 or values.dtype == np.float64:
            return _float_feature_list(values.astype(np.float32))
        else:
            return _int64_feature_list(values.astype(np.int64))

    observations, actions, rewards, done, step_mul = sequence

    context = tf.train.Features(feature={
        'sequence_length': _int64_feature(len(rewards)),
        'score': _float_feature(sum(rewards))
    })

    feature_list = {
        'observation': tf.nest.map_structure(_create_feature, observations),
        'action': tf.nest.map_structure(_create_feature, actions),
        'reward': _float_feature_list(rewards),
        'done': _int64_feature_list(done),
        'step_mul': _int64_feature_list(step_mul),
    }

    feature_list = flatten_nested_dicts(feature_list)

    return tf.train.SequenceExample(context=context, feature_lists=tf.train.FeatureLists(feature_list=feature_list))


def parse_sequence_example(observation_space, action_space, example_proto):
    def _feature_specs(space: spaces.Dict):
        if isinstance(space, spaces.Dict):
            return {k: _feature_specs(v) for k, v in space.spaces.items()}
        elif space.shape and len(space.shape) > 0:
            return tf.FixedLenSequenceFeature((), tf.string)
        else:
            return tf.FixedLenSequenceFeature((), tf.int64)

    def _set_type_and_shape(value: tf.Tensor):
        if value.dtype == tf.int64:
            return tf.cast(value, tf.int32)
        elif value.dtype == tf.float32:
            return value
        elif value.dtype == tf.string:
            return tf.decode_raw(value, tf.uint8)
        else:
            raise ValueError(f"unexpected type {value.dtype.name}")

    context_features = {
        'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
        'score': tf.FixedLenFeature([], dtype=tf.float32),
    }
    sequence_features = {
        'observation': tf.nest.map_structure(_feature_specs, observation_space),
        'action': tf.nest.map_structure(_feature_specs, action_space),
        'reward': tf.FixedLenSequenceFeature((), tf.float32),
        'done': tf.FixedLenSequenceFeature((), tf.int64),
        'step_mul': tf.FixedLenSequenceFeature((), tf.int64),
    }

    sequence_features = flatten_nested_dicts(sequence_features)

    context, parsed_features = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    parsed_features = unflatten_nested_dicts(parsed_features)

    context = tf.nest.map_structure(_set_type_and_shape, context)
    parsed_features = tf.nest.map_structure(_set_type_and_shape, parsed_features)
    parsed_features['observation']['pov'] = tf.reshape(parsed_features['observation']['pov'],
                                                       (-1,) + observation_space['pov'].shape)
    parsed_features['done'] = tf.cast(parsed_features['done'], tf.bool)

    return context, parsed_features


def serialize_single_example(example) -> tf.train.SequenceExample:
    def _create_feature(values: np.ndarray):
        if len(values.shape) > 1:
            return _bytes_feature(values.tobytes())
        elif values.dtype == np.float32 or values.dtype == np.float64:
            return _float_feature(values.astype(np.float32))
        else:
            return _int64_feature(values.astype(np.int64))

    observations, actions, rewards, done = example

    feature_list = {
        'observation': tf.nest.map_structure(_create_feature, observations),
        'action': tf.nest.map_structure(_create_feature, actions),
        'reward': _float_feature(rewards),
        'done': _int64_feature(done)
    }

    feature_list = flatten_nested_dicts(feature_list)

    return tf.train.Example(features=tf.train.Features(feature=feature_list))


def parse_single_example(observation_space, action_space, example_proto):
    def _feature_specs(space: spaces.Dict):
        if isinstance(space, spaces.Dict):
            return {k: _feature_specs(v) for k, v in space.spaces.items()}
        elif space.shape and len(space.shape) > 0:
            return tf.FixedLenFeature((), tf.string)
        else:
            return tf.FixedLenFeature((), tf.int64)

    def _set_type_and_shape(value: tf.Tensor):
        if value.dtype == tf.int64:
            return tf.cast(value, tf.int32)
        elif value.dtype == tf.float32:
            return value
        elif value.dtype == tf.string:
            return tf.decode_raw(value, tf.uint8)
        else:
            raise ValueError(f"unexpected type {value.dtype.name}")

    features = {
        'observation': tf.nest.map_structure(_feature_specs, observation_space),
        'action': tf.nest.map_structure(_feature_specs, action_space),
        'reward': tf.FixedLenFeature((), tf.float32),
        'done': tf.FixedLenFeature((), tf.int64),
    }

    features = flatten_nested_dicts(features)

    context, parsed_features = tf.parse_single_example(example_proto, features)

    parsed_features = unflatten_nested_dicts(parsed_features)

    context = tf.nest.map_structure(_set_type_and_shape, context)
    parsed_features = tf.nest.map_structure(_set_type_and_shape, parsed_features)
    parsed_features['observation']['pov'] = tf.reshape(parsed_features['observation']['pov'],
                                                       observation_space['pov'].shape)
    parsed_features['done'] = tf.cast(parsed_features['done'], tf.bool)

    return context, parsed_features
