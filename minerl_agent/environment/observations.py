import enum
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

import scipy.ndimage
from minerl.env import obtain_observation_space, spaces

from utility.utils import gym_space_to_dict, rgb2gray


class ColorSpace(enum.IntEnum):
    RGB = 0
    GrayScale = 1


class ObservationSpace(ABC):

    @abstractmethod
    def specs(self) -> Dict:
        pass

    @abstractmethod
    def pov_resolution(self) -> int:
        pass

    @abstractmethod
    def pov_color_space(self) -> ColorSpace:
        pass

    @abstractmethod
    def transform_to(self, observation) -> Dict:
        pass

    @abstractmethod
    def transform_from(self, observation) -> Dict:
        pass


class CustomObservationSpace(ObservationSpace):

    def __init__(self, pov_resolution: int = 64, pov_color_space: ColorSpace = ColorSpace.RGB) -> None:
        super().__init__()
        self._pov_resolution = pov_resolution
        self._pov_color_space = pov_color_space

    def pov_resolution(self) -> int:
        return self._pov_resolution

    def pov_color_space(self) -> ColorSpace:
        return self._pov_color_space

    def specs(self) -> Dict:
        specs = gym_space_to_dict(obtain_observation_space)
        if self._pov_resolution != specs['pov'].shape[0]:
            num_channels = 1 if self._pov_color_space == ColorSpace.GrayScale else 3
            new_shape = (self._pov_resolution, self._pov_resolution, num_channels)
            specs['pov'] = spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
        return specs

    def transform_to(self, observation) -> Dict:
        if self._pov_resolution != obtain_observation_space['pov'].shape[0]:
            # resize pov to (self._pov_resolution x self._pov_resolution)
            factor = self._pov_resolution / float(obtain_observation_space['pov'].shape[0])
            if len(observation['pov'].shape) == 3:
                observation['pov'] = scipy.ndimage.zoom(observation['pov'], (factor, factor, 1))
            else:
                observation['pov'] = scipy.ndimage.zoom(observation['pov'], (1, factor, factor, 1))
            observation['pov'] = observation['pov'].astype(obtain_observation_space['pov'].dtype)

        if self._pov_color_space == ColorSpace.GrayScale:
            observation['pov'] = rgb2gray(observation['pov'])
            observation['pov'] = observation['pov'].astype(obtain_observation_space['pov'].dtype)

        if 'inventory' not in observation:
            def _init_with_zero(space):
                if isinstance(space, spaces.Dict):
                    return {k: _init_with_zero(v) for k, v in space.spaces.items() if not k == 'pov'}
                return np.zeros((observation['pov'].shape[0],) + space.shape, space.dtype)
            # Observation is from MineRLTreechop environment
            empty_observation = _init_with_zero(obtain_observation_space)
            empty_observation['pov'] = observation['pov']
            return empty_observation
        return observation

    def transform_from(self, observation) -> Dict:
        raise NotImplementedError
