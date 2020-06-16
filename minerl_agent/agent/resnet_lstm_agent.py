import functools
from typing import List

import sonnet as snt
import tensorflow as tf

from minerl_agent.agent.agent import Agent, EmbedType, inputs_spatial, inputs_non_spatial, AgentOutput
from minerl_agent.environment.actions import ActionSpace
from minerl_agent.environment.observations import ObservationSpace
from utility.utils import gym_space_to_dict

DEFAULT_CRAFT_OBSERVATIONS = [
    "coal", "cobblestone", "crafting_table", "furnace", "iron_ingot", "iron_ore", "log", "planks", "stick", "stone",
    "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe"
]


def process_inputs_craft(observations: dict, craft_inputs: List[str]):
    craft_obs = [
        tf.expand_dims(tf.log(tf.to_float(tf.maximum(observations['inventory'][item_name] + 1, 1))), axis=-1)
        for item_name in craft_inputs
    ]
    return tf.concat(craft_obs, axis=-1)


class ResnetLSTMAgent(Agent):

    def __init__(self, observation_space: ObservationSpace, action_space: ActionSpace, max_step_mul: int,
                 core_hidden_size=256, use_prev_actions: bool = False, action_embed_type: EmbedType = EmbedType.EMBED,
                 action_embed_size=16, craft_observation_type: List[str] = None) -> None:
        super(ResnetLSTMAgent, self).__init__(name='resnet_lstm_agent')
        self._observation_space = observation_space
        self._observation_specs = gym_space_to_dict(observation_space.specs())
        self._action_space = action_space
        self._action_specs = gym_space_to_dict(action_space.specs())
        self._max_step_mul = max_step_mul
        self._core_hidden_size = core_hidden_size
        self._use_prev_actions = use_prev_actions
        self._action_embed_type = action_embed_type
        self._action_embed_size = action_embed_size
        self._craft_inputs = craft_observation_type or DEFAULT_CRAFT_OBSERVATIONS
        self._craft_actions = ['craft', 'nearbyCraft', 'nearbySmelt']
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(self._core_hidden_size)

    @property
    def state_size(self):
        return self._core.state_size

    @property
    def output_size(self):
        return AgentOutput(
            tf.nest.map_structure(lambda s: tf.TensorShape([]), self._action_specs),
            tf.nest.map_structure(lambda s: tf.TensorShape([s.n]), self._action_specs),
            tf.TensorShape([]),
            tf.TensorShape([self._max_step_mul]),
            tf.TensorShape([]),
        )

    def initial_state(self, batch_size, **unused_kwargs):
        return self._core.zero_state(batch_size, tf.float32)

    def _torso(self, inputs):
        actions, rewards, done, observations = inputs

        # process spatial observations by residual conv layers
        with tf.variable_scope('inputs_spatial'):
            spatial_inputs = inputs_spatial(observations)
            if self._observation_specs['pov'].shape[0] >= 32:
                blocks = [(16, 2), (32, 2), (32, 2)]
            elif self._observation_specs['pov'].shape[0] >= 16:
                blocks = [(16, 2), (32, 4)]
            else:
                blocks = [(32, 6)]
            for i, (num_ch, num_blocks) in enumerate(blocks):
                # Downscale.
                spatial_inputs = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(spatial_inputs)
                spatial_inputs = tf.nn.pool(spatial_inputs, window_shape=[3, 3], pooling_type='MAX', padding='SAME',
                                            strides=[2, 2])

                # Residual block(s).
                for j in range(num_blocks):
                    with tf.variable_scope('residual_%d_%d' % (i, j)):
                        block_input = spatial_inputs
                        spatial_inputs = tf.nn.relu(spatial_inputs)
                        spatial_inputs = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(spatial_inputs)
                        spatial_inputs = tf.nn.relu(spatial_inputs)
                        spatial_inputs = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(spatial_inputs)
                        spatial_inputs += block_input

            spatial_inputs = tf.nn.relu(spatial_inputs)
            spatial_inputs = snt.BatchFlatten()(spatial_inputs)
            spatial_inputs = snt.Linear(256)(spatial_inputs)
            spatial_inputs = tf.nn.relu(spatial_inputs)

        with tf.variable_scope('inputs_non_spatial'):
            non_spatial_inputs = inputs_non_spatial(observations)
            if self._use_prev_actions:
                if self._action_embed_type == EmbedType.EMBED:
                    action_embeddings = [
                        snt.Embed(vocab_size=self._action_specs[action_name].n, embed_dim=self._action_embed_size)(
                            action[:, 0] if len(action.get_shape()) >= 2 else action)
                        for action_name, action in sorted(actions.items())
                    ]
                else:
                    action_embeddings = [
                        tf.one_hot(indices=action[:, 0] if len(action.get_shape()) >= 2 else action,
                                   depth=self._action_specs[action_name].n)
                        for action_name, action in sorted(actions.items())
                    ]
                non_spatial_inputs = tf.concat([non_spatial_inputs] + action_embeddings, axis=-1)
            non_spatial_inputs = snt.Linear(256)(non_spatial_inputs)
            non_spatial_inputs = tf.nn.relu(non_spatial_inputs)
            non_spatial_inputs = snt.Linear(64)(non_spatial_inputs)
            non_spatial_inputs = tf.nn.relu(non_spatial_inputs)

        with tf.variable_scope('inputs_craft'):
            craft_inputs = process_inputs_craft(observations, self._craft_inputs)
            craft_inputs = snt.Linear(256)(craft_inputs)
            craft_inputs = tf.nn.relu(craft_inputs)
            craft_inputs = snt.Linear(64)(craft_inputs)
            craft_representation = tf.nn.relu(craft_inputs)

        representation = tf.concat([spatial_inputs, non_spatial_inputs], axis=-1)
        return representation, craft_representation

    def _head(self, inputs, inputs_craft):
        # baseline function:
        with tf.variable_scope('baseline_head'):
            baseline = inputs
            baseline = snt.Linear(256)(baseline)
            baseline = tf.nn.relu(baseline)
            baseline = tf.squeeze(snt.Linear(1, name='baseline')(baseline), axis=-1)

        # action policy:
        with tf.variable_scope('policy_head'):
            actions, action_logits, action_embeddings = {}, {}, {}
            for action_name, action_specs in self._action_specs.items():
                with tf.variable_scope(action_name):
                    if action_name in self._craft_actions:
                        inputs = tf.concat([inputs, inputs_craft], axis=-1)
                    action_logits[action_name] = snt.Linear(action_specs.n)(inputs)
                    actions[action_name] = tf.random.categorical(logits=action_logits[action_name], num_samples=1,
                                                                 dtype=tf.int32)
        # step mul policy:
        with tf.variable_scope('step_mul_head'):
            step_mul_logits = snt.Linear(self._max_step_mul)(inputs)
            step_mul = tf.multinomial(logits=step_mul_logits, num_samples=1, output_dtype=tf.int32)

        return AgentOutput(actions, action_logits, step_mul, step_mul_logits, baseline)

    def _build(self, input_, core_state, dynamic_unroll=False, sequence_length=None):
        action, env_output = input_
        actions, env_outputs = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state, dynamic_unroll, sequence_length)
        return tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state, dynamic_unroll=False, sequence_length=None):
        if len(env_outputs) == 4:
            rewards, _, done, observations = env_outputs
        else:
            rewards, done, observations = env_outputs

        if isinstance(actions, list):
            actions = tf.nest.pack_sequence_as(self._action_specs, actions)

        if isinstance(observations, list):
            observations = tf.nest.pack_sequence_as(self._observation_specs, observations)

        representation, craft_representation = snt.BatchApply(self._torso)((actions, rewards, done, observations))

        if dynamic_unroll:
            initial_core_state = self._core.zero_state(tf.shape(done)[1], tf.float32)
            core_output, core_state = tf.nn.dynamic_rnn(self._core, representation, sequence_length=sequence_length,
                                                        initial_state=initial_core_state, time_major=True,
                                                        scope='rnn')
        else:
            # to be compatible with dynamic_rnn unroll, we must enter the same variable scope here
            with tf.variable_scope('rnn'):
                initial_core_state = self._core.zero_state(tf.shape(done)[1], tf.float32)
                core_output_list = []
                for input_, d in zip(tf.unstack(representation), tf.unstack(done)):
                    # If the episode ended, the core state should be reset before the next.
                    core_state = tf.nest.map_structure(functools.partial(tf.where, d), initial_core_state, core_state)
                    core_output, core_state = self._core(input_, core_state)
                    core_output_list.append(core_output)
                core_output = tf.stack(core_output_list)

        return snt.BatchApply(self._head)(core_output, craft_representation), core_state
