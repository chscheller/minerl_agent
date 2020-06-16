import logging
from abc import ABC, abstractmethod
from typing import Dict, Set

import numpy as np
import tensorflow as tf
from minerl.env import spaces, obtain_action_space

logger = logging.getLogger(__name__)


class ActionSpace(ABC):

    @abstractmethod
    def num_camera_actions(self) -> int:
        pass

    @abstractmethod
    def camera_max_angle(self) -> int:
        pass

    @abstractmethod
    def specs(self) -> Dict:
        pass

    @abstractmethod
    def transform(self, action) -> Dict:
        pass

    @abstractmethod
    def transform_back(self, action) -> Dict:
        pass

    @abstractmethod
    def unimportant_actions(self) -> Set[str]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class CustomActionSpace(ActionSpace):
    def __init__(self, num_camera_actions: int = 11, camera_max_angle: int = 180) -> None:
        super().__init__()
        self._num_camera_actions = num_camera_actions
        self._camera_max_angle = camera_max_angle

    @property
    def name(self) -> str:
        return 'V2'

    def num_camera_actions(self) -> int:
        return self._num_camera_actions

    def camera_max_angle(self) -> int:
        return self._camera_max_angle

    def specs(self) -> Dict:
        return {
            "forward": spaces.Discrete(2),
            "jump": spaces.Discrete(2),
            "camera_pitch": spaces.Discrete(self._num_camera_actions),
            "camera_yaw": spaces.Discrete(self._num_camera_actions),
            "place": obtain_action_space["place"],
            "equip": obtain_action_space["equip"],
            "craft": obtain_action_space["craft"],
            "nearbyCraft": obtain_action_space["nearbyCraft"],
            "nearbySmelt": obtain_action_space["nearbySmelt"]
        }

    def unimportant_actions(self) -> Set[str]:
        return {'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack'}

    def transform(self, action) -> Dict:
        transformed_action = {}
        camera = np.asarray(action['camera'])
        for name, space in self.specs().items():
            if name == 'camera_pitch':
                # spaces.n - 1 since we allow 0:
                transformed_action[name] = np.int32(
                    np.round((space.n - 1) * (camera[:, 0] + self._camera_max_angle) / (2 * self._camera_max_angle)))
                assert np.alltrue(transformed_action[name] >= 0)
                assert np.alltrue(transformed_action[name] < space.n)
            elif name == 'camera_yaw':
                transformed_action[name] = np.int32(
                    np.round((space.n - 1) * (camera[:, 1] + self._camera_max_angle) / (2 * self._camera_max_angle)))
                assert np.alltrue(transformed_action[name] >= 0)
                assert np.alltrue(transformed_action[name] < space.n)
            elif name in action.keys():
                transformed_action[name] = action[name]
            else:
                transformed_action[name] = np.zeros((action['forward'].shape[0],) + space.shape, space.dtype)
        return transformed_action

    def transform_treechop(self, action) -> Dict:
        pass

    def transform_back(self, action) -> Dict:
        action_spec = self.specs()
        if isinstance(action['forward'], int) or action['forward'].size == 1:
            # For single actions we squeeze the action values and transform them to python int. This makes
            # them compatible to use directly with the minerl env.
            transformed_action = {
                'camera': (
                    int(np.round((2 * self._camera_max_angle * np.squeeze(action['camera_pitch']) / (
                            action_spec['camera_pitch'].n - 1)) - self._camera_max_angle)),
                    int(np.round((2 * self._camera_max_angle * np.squeeze(action['camera_yaw']) / (
                            action_spec['camera_yaw'].n - 1)) - self._camera_max_angle)),
                ),
                'attack': 1,
                'back': 0,
                'left': 0,
                'right': 0,
                'sprint': 0,
                'sneak': 0
            }
            for name, value in action.items():
                if name not in {'camera_pitch', 'camera_yaw'}:
                    transformed_action[name] = int(np.squeeze(value))
            return transformed_action
        else:
            transformed_action = {
                'camera': np.stack((
                    np.round((2 * self._camera_max_angle * np.squeeze(action['camera_pitch']) / (
                            action_spec['camera_pitch'].n - 1)) - self._camera_max_angle),
                    np.round((2 * self._camera_max_angle * np.squeeze(action['camera_yaw']) / (
                            action_spec['camera_yaw'].n - 1)) - self._camera_max_angle),
                ), axis=-1),
                'attack': np.ones_like(action['forward']),
                'back': np.zeros_like(action['forward']),
                'left': np.zeros_like(action['forward']),
                'right': np.zeros_like(action['forward']),
                'sprint': np.zeros_like(action['forward']),
                'sneak': np.zeros_like(action['forward'])
            }
            for name, value in action.items():
                if name not in {'camera_pitch', 'camera_yaw'}:
                    transformed_action[name] = value
            return transformed_action


def skip_action_repeats_accumulate_camera(sequence, max_step_mul, camera_max_angle, unimportant_actions: Set[str],
                                          ignore_unimportant_actions=False):
    """ Subsamples sequences by frame skipping, camera accumulation and truncation. All operations are
    independent of game state in order to be compliant with the minerl competition rules.

    The following operations are applied:
        - Frames with all-zero actions (no_op) are skipped
        - Succeeding frames that contain the same actions are skipped for up to `max_step_mul` steps and instead a step
          multiplier is introduced.
        - Camera actions are accumulated until a certain threshold is reached (`camera_max_angle`) or a new action is
          issued.

    Reducing the sequence length is crucial in order to be able to train LSTMS on full episodes, that can become
    very long (up to 180000 frames).

    Args:
        sequence (tuple): A (s, a, r, d) tuple where each field is of shape [T, ...].
        max_step_mul (int): The maximum number of steps that an action can be
        camera_max_angle (int): The maximum angle in either direction (sign ignored)
        unimportant_actions (Set[str]): A set of action names that should not be considered when checking for
            same/idle actions.
        ignore_unimportant_actions: Whether to ignore the given set of unimportant actions when checking for
            same/idle actions or not.

    Returns:
        A (s, a, r, d, step_mul) tuple where each field is of shape [T', ...] where T' <= T
    """
    def are_actions_same(t1, t2):
        return np.all([is_action_same(action_name, action[t1], action[t2]) for action_name, action in actions.items()])

    def is_action_same(action_name, a1, a2):
        if ignore_unimportant_actions and action_name in unimportant_actions:
            return True
        elif action_name == 'camera':
            return np.all(np.sign(a1) == np.sign(a2))
        else:
            return np.all(a1 == a2)

    def all_actions_idle(t) -> bool:
        return np.all([is_idle_action(action_name, action, rewards[t], t) for action_name, action in actions.items()])

    def is_idle_action(action_name, action, reward, t) -> bool:
        if ignore_unimportant_actions and action_name in unimportant_actions:
            return reward == 0
        elif action_name == 'camera':
            if not np.all(action[t] == [0, 0]):
                return False
        else:
            if not action[t] == 0:
                return False
        return reward == 0

    states, actions, rewards, _ = sequence

    actions['camera'] = np.asarray(actions['camera'])

    sequence_length = len(actions['forward'])
    if 'nearbyCraft' not in actions:
        # Treechop environment -> end sequence after first reward
        reward_indexes = np.where(rewards == 1)[0]
        sequence_length = min(sequence_length, reward_indexes[0] + 1 if 1 in rewards else sequence_length)

    prev_t = 0
    selected_indices = [0]
    # step_mul, reward and camera are accumulated
    step_mul = []
    accumulated_rewards = []
    accumulated_camera = []
    current_reward = rewards[0]
    current_camera = actions['camera'][0]
    skipped_steps = 0

    for t in range(1, sequence_length):
        # skip idle actions
        if all_actions_idle(t):
            skipped_steps += 1
            continue

        # An accumulated step is complete as soon as one of the following conditions is met:
        #   - The action at the current time step differs from the previous one (camera rotations with the same sign
        #     are considered equal!)
        #   - `max_step_mul` is reached for the current accumulation
        #   - The accumulated camera rotation would exceed [-camera_max_angle, camera_max_angle] when adding the
        #     camera rotation at the current time step
        #   - One or more frames were skipped due to the demonstrator being idle
        if not are_actions_same(t, prev_t) \
                or (t - prev_t == max_step_mul) \
                or np.any(np.abs(current_camera + actions['camera'][t]) > camera_max_angle) \
                or skipped_steps > 0:
            selected_indices.append(t)
            step_mul.append(t - prev_t - skipped_steps)
            accumulated_rewards.append(current_reward)
            accumulated_camera.append(np.clip(current_camera, -camera_max_angle, camera_max_angle))
            current_reward = 0
            current_camera = np.zeros_like(current_camera)
            skipped_steps = 0
            prev_t = t

        current_reward += rewards[t]
        current_camera += actions['camera'][t]

    # still need to add the currently accumulated values:
    step_mul.append(sequence_length - prev_t - skipped_steps)
    accumulated_rewards.append(current_reward)
    accumulated_camera.append(np.clip(current_camera, -camera_max_angle, camera_max_angle))

    states, actions = tf.nest.map_structure(lambda x: x[selected_indices], (states, actions))
    step_muls = np.asarray(step_mul) - 1
    actions['camera'] = np.asarray(accumulated_camera)
    rewards = np.asarray(accumulated_rewards)
    done = np.full(rewards.shape, False, dtype=np.bool)
    done[-1] = True

    # some assertions to verify consistency
    lengths = np.asarray([len(x) for x in (selected_indices, rewards, done, step_muls)])
    assert np.all(step_muls >= 0), f"step_mul {np.min(step_muls)} >= 0"
    assert np.all(step_muls < max_step_mul), f"step_mul {np.max(step_muls)} < {max_step_mul}"
    assert np.all(lengths == len(selected_indices)), f'all lengths same: {lengths}, {len(selected_indices)}'
    assert np.all([not all_actions_idle(t) for t in
                   range(1, len(selected_indices))]), f'no action idle'  # note: first action might be idle!
    assert np.all(np.abs(actions['camera']) <= camera_max_angle), \
        f"camera {-camera_max_angle}<={np.min(actions['camera'])}<={np.max(actions['camera'])}<={camera_max_angle}"

    return states, actions, rewards, done, step_muls
