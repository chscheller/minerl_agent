# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments and environment helper classes."""

import collections
import enum
import traceback

import gym
import numpy as np
import tensorflow as tf
from minerl.env import spaces, malmo

from minerl_agent.environment.actions import ActionSpace
from minerl_agent.environment.observations import ObservationSpace

nest = tf.nest


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


class PyProcessMineRLObtain(object):
    def __init__(self, config, seed):
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._launch_lock = config['launch_lock']
        self._action_space = config['action_space']  # type: ActionSpace
        self._action_specs = self._action_space.specs()
        self._observation_space = config['observation_space']  # type: ObservationSpace
        self._observation_specs = self._observation_space.specs()
        self._fixed_step_mul = config['fixed_step_mul']
        self._step_mul = config['step_mul']
        self._max_episode_length = config['max_episode_length']
        self._num_env_frames = config['num_env_frames']
        self._num_env_frames_lock = config['num_env_frames_lock']
        self._malmo_base_port = config['malmo_base_port']
        self._environment = config['environment_name']
        self._state = StepType.LAST
        self._episode_steps = 0
        if self._malmo_base_port is not None:
            malmo.InstanceManager.configure_malmo_base_port(self._malmo_base_port)
        self._launch()

    def _launch(self):
        with self._launch_lock:
            tf.logging.info(f"Starting {self._environment} environment.")
            self._env = gym.make(self._environment)
            if self._seed is not None:
                self._env.seed(self._seed)
                self._seed = None  # in case of a env restart, we don't want to use the same seed again
            tf.logging.info(f"{self._environment} environment started successfully...")

    def _restart(self):
        tf.logging.info(f"Restart {self._environment} environment.")
        try:
            self._env.close()
        except Exception as e:
            tf.logging.error('Failed to close environment (stacktrace below). Ignore the error for now.')
            traceback.print_exc()
        self._launch()

    def _reset(self):
        self._state = StepType.FIRST
        self._episode_steps = 0
        return self._env.reset()

    def _transform_observation(self, obs):
        obs['equipped_items']['mainhand'] = {
            'damage': obs['equipped_items']['mainhand'].get('damage', 0),
            'maxDamage': obs['equipped_items']['mainhand'].get('maxDamage', 0),
            'type': obs['equipped_items']['mainhand']['type']
            if isinstance(obs['equipped_items']['mainhand']['type'], int)
            else self._observation_specs['equipped_items']['mainhand']['type'].n - 1
        }
        obs = self._observation_space.transform_to(obs)
        return [np.squeeze(x).astype(np.int32 if isinstance(x, int) or x.dtype == np.int64 else x.dtype)
                for x in nest.flatten(obs)]

    def initial(self):
        obs = self._reset()
        return self._transform_observation(obs)

    def _step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._episode_steps += 1
        if self._max_episode_length.value > 0 and self._episode_steps >= self._max_episode_length.value:
            done = True
        with self._num_env_frames_lock:
            self._num_env_frames.value += 1
        return obs, reward, done

    def step(self, raw_action, step_mul):
        """
        Args:
            raw_action: flattened custom action
            step_mul: step multiplier
        Returns:
            A (reward, done, observation) tuple.
            Be aware that done is true for the FIRST frame of each episode only.
        """
        try:
            if self._state == StepType.LAST:
                return np.float32(0.), True, self.initial()
            if self._fixed_step_mul:
                # if step_mul is fixed, step_mul from the actor is overridden by the fixed step_mul
                step_mul = self._step_mul - 1
            self._state = StepType.MID
            action = nest.pack_sequence_as(self._action_specs, raw_action.tolist())
            action = {key: int(np.squeeze(value)) for key, value in action.items()}
            gym_action = self._action_space.transform_back(action)
            gym_action['camera'] = np.asarray(gym_action['camera']) / (float(np.squeeze(step_mul)) + 1)
            obs, total_reward, done = self._step(gym_action)
            if self._fixed_step_mul:
                # if step_mul is fixed, we need to reset "one_step" actions after the first step:
                for action_name in ["jump", "place", "craft", "nearbyCraft", "nearbySmelt"]:
                    gym_action[action_name] = 0
            for j in range(int(np.squeeze(step_mul))):
                if done:
                    break
                obs, reward, done = self._step(gym_action)
                total_reward += reward
            if done:
                self._state = StepType.LAST
            return np.float32(np.squeeze(total_reward)), False, self._transform_observation(obs)
        except Exception as e:
            tf.logging.error('Failed to take environment step (stacktrace below). Try to reset/restart.')
            traceback.print_exc()
            try:
                tf.logging.info('Reset environment after exception.')
                return np.float32(0.), True, self.initial()
            except Exception as e:
                tf.logging.error('Failed to reset environment after exception (stacktrace below). Try to restart.')
                traceback.print_exc()
                self._restart()
            return np.float32(0.), True, self.initial()

    def close(self):
        self._env.close()

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""
        config = constructor_kwargs['config']
        observation_space = config['observation_space']  # type: ObservationSpace

        observation_spec = [
            tf.contrib.framework.TensorSpec(
                [] if s.dtype == spaces.Enum or len(s.shape) == 0 or s.shape[0] <= 1 else list(s.shape),
                tf.as_dtype(np.int32 if s.dtype == np.int64 else s.dtype)
            )
            for s in nest.flatten(observation_space.specs())
        ]

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            return (
                tf.contrib.framework.TensorSpec([], tf.float32),
                tf.contrib.framework.TensorSpec([], tf.bool),
                observation_spec,
            )


StepOutputInfo = collections.namedtuple('StepOutputInfo', 'episode_return episode_step environment_id')
StepOutput = collections.namedtuple('StepOutput', 'reward info done observation')


class FlowEnvironment(object):
    """An environment that returns a new state for every modifying method.

    The environment returns a new environment state for every modifying action and
    forces previous actions to be completed first. Similar to `flow` for
    `TensorArray`.
    """

    def __init__(self, env, env_id):
        """Initializes the environment.

        Args:
          env: An environment with `initial()` and `step(action)` methods where
            `initial` returns the initial observations and `step` takes an action
            and returns a tuple of (reward, done, observation). `observation`
            should be the observation after the step is taken. If `done` is
            True, the observation should be the first observation in the next
            episode.
          env_id: unique environment identifier
        """
        self._env = env
        self._env_id = env_id

    def initial(self):
        """Returns the initial output and initial state.

        Returns:
          A tuple of (`StepOutput`, environment state). The environment state should
          be passed in to the next invocation of `step` and should not be used in
          any other way. The reward and transition type in the `StepOutput` is the
          reward/transition type that lead to the observation in `StepOutput`.
        """
        with tf.name_scope('flow_environment_initial'):
            initial_reward = tf.constant(0.)
            initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0), tf.constant(self._env_id))
            initial_done = tf.constant(True)
            initial_observation = self._env.initial()

            initial_output = StepOutput(
                initial_reward,
                initial_info,
                initial_done,
                initial_observation)

            # Control dependency to make sure the next step can't be taken before the
            # initial output has been read from the environment.
            with tf.control_dependencies(nest.flatten(initial_output)):
                initial_flow = tf.constant(0, dtype=tf.int64)
            initial_state = (initial_flow, initial_info)
            return initial_output, initial_state

    def step(self, action, step_mul, state):
        """Takes a step in the environment.

        Args:
          action: An action tensor suitable for the underlying environment.
          state: The environment state from the last step or initial state.

        Returns:
          A tuple of (`StepOutput`, environment state). The environment state should
          be passed in to the next invocation of `step` and should not be used in
          any other way. On episode end (i.e. `done` is True), the returned reward
          should be included in the sum of rewards for the ending episode and not
          part of the next episode.
        """
        with tf.name_scope('flow_environment_step'):
            flow, info = nest.map_structure(tf.convert_to_tensor, state)

            # Make sure the previous step has been executed before running the next
            # step.
            with tf.control_dependencies([flow]):
                reward, done, observation = self._env.step(action, step_mul)

            with tf.control_dependencies(nest.flatten(observation)):
                new_flow = tf.add(flow, 1)

            # When done, include the reward in the output info but not in the
            # state for the next step.
            new_info = StepOutputInfo(info.episode_return + reward,
                                      info.episode_step + 1,
                                      info.environment_id)
            new_state = new_flow, nest.map_structure(
                lambda a, b: tf.where(done, a, b),
                StepOutputInfo(tf.constant(0.), tf.constant(0), info.environment_id),
                new_info)

            output = StepOutput(reward, new_info, done, observation)
            return output, new_state
