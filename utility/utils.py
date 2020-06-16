import collections
from abc import abstractmethod, ABC

import numpy as np
from typing import Optional

import gym
import tensorflow as tf


def accuracy(labels: tf.Tensor, predictions: tf.Tensor, sequence_lengths: Optional[tf.Tensor] = None) -> tf.Tensor:
    if sequence_lengths is not None:
        if len(labels.get_shape()) != 2:
            raise ValueError('Expect labels of rank {} but rank {} given.'.format(2, len(labels.get_shape())))
        if len(predictions.get_shape()) != 2:
            raise ValueError('Expect predictions of rank {} but rank {} given.'.format(2, len(predictions.get_shape())))
        if len(sequence_lengths.get_shape()) != 1:
            raise ValueError('Expect sequence_lengths of rank {} but rank {} given.'
                             .format(1, len(sequence_lengths.get_shape())))
        sequence_mask = swap_leading_axes(tf.sequence_mask(sequence_lengths, dtype=tf.float32))
        correct_predictions = tf.cast(tf.equal(labels, tf.cast(predictions, labels.dtype)), tf.float32) * sequence_mask
        correct_predictions = tf.reduce_sum(correct_predictions, axis=0)
        sequence_accuracies = correct_predictions / tf.cast(sequence_lengths, tf.float32)
        return tf.reduce_mean(sequence_accuracies)
    else:
        return tf.reduce_mean(tf.cast(tf.equal(labels, tf.cast(predictions, labels.dtype)), tf.float32))


def swap_leading_axes(tensor: tf.Tensor) -> tf.Tensor:
    return tf.transpose(tensor, perm=[1, 0] + list(range(2, len(tensor.get_shape()))))


def gym_space_to_dict(space: gym.spaces.space):
    if isinstance(space, gym.spaces.Dict):
        return {key: gym_space_to_dict(value) for key, value in space.spaces.items()}
    else:
        return space


def flatten_nested_dicts(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dicts(d, sep='/'):
    out_dict = {}
    for k, v in d.items():
        dict_pointer = out_dict
        key_path = list(k.split(sep))
        if len(key_path) > 1:
            for sub_k in key_path[:-1]:
                if sub_k not in dict_pointer:
                    dict_pointer[sub_k] = {}
                dict_pointer = dict_pointer[sub_k]
        dict_pointer[key_path[-1]] = v
    return out_dict


def rgb2gray(rgb):
    assert len(rgb.shape) >= 3
    assert rgb.shape[-1] == 3
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)
