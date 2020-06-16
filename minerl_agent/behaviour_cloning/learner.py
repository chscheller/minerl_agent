import logging

import tensorflow as tf

from minerl_agent.agent.agent import Agent
from utility.utils import swap_leading_axes, accuracy

logger = logging.getLogger(__name__)


def build_learner(agent: Agent, learner_inputs: tuple, learning_rate: float, adam_beta1=0.9, adam_beta2=0.999,
                  adam_epsilon=1e-8, clip_grad_norm=100) -> tf.Operation:
    context, features = learner_inputs

    # we need to swap leading axes since the agent expects sequence first
    features = tf.nest.map_structure(swap_leading_axes, features)  # B x T -> T x B
    prev_actions = tf.nest.map_structure(lambda a: tf.pad(a[:-1], [[1, 0], [0, 0]], "CONSTANT"), features['action'])
    env_outputs = (features['reward'], features['done'], features['observation'])
    initial_agent_state = agent.initial_state(tf.shape(features['reward'])[1])

    # unroll agent
    learner_outputs, _ = agent.unroll(prev_actions, env_outputs, core_state=initial_agent_state,
                                      dynamic_unroll=True, sequence_length=context['sequence_length'])

    # mask padding at the end of the sequences
    mask = tf.sequence_mask(lengths=context['sequence_length'], maxlen=tf.shape(features['reward'])[0],
                            dtype=tf.float32)

    # policy loss
    neg_log_probs = []
    for action_name, action_logits in learner_outputs.action_logits.items():
        neg_log_probs.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=features['action'][action_name],
                                                                            logits=action_logits))
        tf.summary.scalar(f'accuracy/{action_name}', accuracy(features['action'][action_name],
                                                              tf.squeeze(learner_outputs.action[action_name], axis=2),
                                                              context['sequence_length']))
    neg_log_probs.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=features['step_mul'],
                                                                        logits=learner_outputs.step_mul_logits))
    tf.summary.scalar(f'accuracy/step_mul', accuracy(features['step_mul'], tf.squeeze(learner_outputs.step_mul, axis=2),
                                                     context['sequence_length']))

    neg_log_probs = swap_leading_axes(sum(neg_log_probs))  # T x B -> B x T
    policy_loss = tf.reduce_mean(neg_log_probs * mask)

    # init optimizer & train op
    total_loss = policy_loss

    optimizer = tf.train.AdamOptimizer(learning_rate, adam_beta1, adam_beta2, adam_epsilon)
    parameters = tf.trainable_variables()
    gradients = tf.gradients(total_loss, parameters)
    if clip_grad_norm > 0:
        gradients, grad_norm = tf.clip_by_global_norm(gradients, clip_grad_norm)
    else:
        grad_norm = tf.global_norm(gradients)
    tf.summary.scalar('grad_norm', grad_norm)

    train_op = optimizer.apply_gradients(list(zip(gradients, parameters)))

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        num_batch_env_frames = tf.to_int64(tf.reduce_sum(context['sequence_length']))
        num_env_frames = tf.train.get_global_step()
        num_env_frames_and_train = num_env_frames.assign_add(num_batch_env_frames)

    # Adding a few summaries.
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss/policy', policy_loss)
    tf.summary.scalar('loss/total', total_loss)
    tf.summary.scalar('max_sequence_length', tf.reduce_max(context['sequence_length']))

    return num_env_frames_and_train
