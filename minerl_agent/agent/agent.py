import collections
import enum
from abc import ABC, abstractmethod

import sonnet as snt
import tensorflow as tf


AgentOutput = collections.namedtuple('AgentOutput', [
    'action', 'action_logits', 'step_mul', 'step_mul_logits', 'baseline'
])


class Agent(snt.RNNCore, ABC):
    @abstractmethod
    def unroll(self, *args, **kwargs):
        pass


class EmbedType(enum.IntEnum):
    EMBED = 0,
    ONE_HOT = 1


def inputs_spatial(observations: dict):
    return tf.to_float(observations['pov']) * (1/255.)


def inputs_non_spatial(observations: dict):
    non_spatial_obs = []
    for key, value in observations.items():
        if key == 'equipped_items':
            non_spatial_obs.append(tf.one_hot(tf.to_int32(value['mainhand']['type']), depth=6, dtype=tf.float32))
            non_spatial_obs.append(tf.expand_dims(tf.to_float(value['mainhand']['damage']) / 1562., axis=-1))
            non_spatial_obs.append(tf.expand_dims(tf.to_float(value['mainhand']['maxDamage']) / 1562., axis=-1))
        elif key == 'inventory':
            non_spatial_obs.extend([
                tf.expand_dims(tf.log(tf.to_float(tf.maximum(inventory_item + 1, 1))), axis=-1)
                for inventory_item in value.values()
            ])
    return tf.concat(non_spatial_obs, axis=-1)


class SeparateActorCriticWrapperAgent(Agent):

    def __init__(self, actor: Agent, critic: Agent) -> None:
        super(SeparateActorCriticWrapperAgent, self).__init__(name='separate_actor_critic_wrapper_agent')
        self._actor = actor
        self._critic = critic

    @property
    def state_size(self):
        return self._actor.state_size, self._critic.state_size

    @property
    def output_size(self):
        return AgentOutput(
            action=self._actor.output_size.action,
            action_logits=self._actor.output_size.action_logits,
            step_mul=self._actor.output_size.step_mul,
            step_mul_logits=self._actor.output_size.step_mul_logits,
            baseline=self._critic.output_size.baseline
        )

    def initial_state(self, batch_size, **unused_kwargs):
        return self._actor.initial_state(batch_size), self._critic.initial_state(batch_size)

    def _build(self, input_, core_state, dynamic_unroll=False, sequence_length=None):
        actor_outputs, actor_core_state = self._actor(input_, core_state[0], dynamic_unroll, sequence_length)
        critic_outputs, critic_core_state = self._critic(input_, core_state[1], dynamic_unroll, sequence_length)
        return AgentOutput(
            action=actor_outputs.action,
            action_logits=actor_outputs.action_logits,
            step_mul=actor_outputs.step_mul,
            step_mul_logits=actor_outputs.step_mul_logits,
            baseline=critic_outputs.baseline
        ), (actor_core_state, critic_core_state)

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state, dynamic_unroll=False, sequence_length=None):
        actor_outputs, actor_core_state = self._actor.unroll(
            actions, env_outputs, core_state[0], dynamic_unroll, sequence_length)
        critic_outputs, critic_core_state = self._critic.unroll(
            actions, env_outputs, core_state[1], dynamic_unroll, sequence_length)
        return AgentOutput(
            action=actor_outputs.action,
            action_logits=actor_outputs.action_logits,
            step_mul=actor_outputs.step_mul,
            step_mul_logits=actor_outputs.step_mul_logits,
            baseline=critic_outputs.baseline
        ), (actor_core_state, critic_core_state)
