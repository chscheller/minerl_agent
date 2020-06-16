import sonnet as snt
import tensorflow as tf

from minerl_agent.impala import vtrace


def clip_rewards(rewards, reward_clipping: str):
    if reward_clipping == 'abs_one':
        return tf.clip_by_value(rewards, -1, 1)
    elif reward_clipping == 'soft_asymmetric':
        squeezed = tf.tanh(rewards / 5.0)
        return tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.
    else:
        return rewards


def clip_gradients(gradients, clip_grad_norm):
    if clip_grad_norm > 0:
        return tf.clip_by_global_norm(gradients, clip_grad_norm)
    else:
        return gradients, tf.global_norm(gradients)


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -entropy_per_timestep


def compute_policy_cloning_loss(logits, previouse_logits):
    log_policy = tf.nn.log_softmax(logits)
    prev_log_policy = tf.nn.log_softmax(previouse_logits)
    prev_policy = tf.nn.softmax(previouse_logits)
    return tf.reduce_sum(prev_policy * (prev_log_policy - log_policy))


def build_learner(agent: snt.RNNCore, agent_state, env_outputs, agent_outputs, reward_clipping: str, discounting: float,
                  baseline_cost: float, entropy_cost: float, policy_cloning_cost: float, value_cloning_cost: float,
                  clip_grad_norm: float, clip_advantage: bool, learning_rate: float, batch_size: int,
                  batch_size_from_replay: int, unroll_length: int, reward_scaling: float = 1.0, adam_beta1: float = 0.9,
                  adam_beta2: float = 0.999, adam_epsilon: float = 1e-8, fixed_step_mul: bool = False,
                  step_mul: int = 8):
    """Builds the learner loop.

    Returns:
        A tuple of (done, infos, and environment frames) where
        the environment frames tensor causes an update.

    """
    learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs, agent_state)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.
    agent_outputs = tf.nest.map_structure(lambda t: t[1:], agent_outputs)
    agent_outputs_from_buffer = tf.nest.map_structure(lambda t: t[:, :batch_size_from_replay], agent_outputs)
    learner_outputs_from_buffer = tf.nest.map_structure(lambda t: t[:-1, :batch_size_from_replay], learner_outputs)

    rewards, infos, done, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
    learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

    rewards = rewards * reward_scaling
    clipped_rewards = clip_rewards(rewards, reward_clipping)
    discounts = tf.to_float(~done) * discounting

    # We only need to learn a step_mul policy if the step multiplier is not fixed.
    if not fixed_step_mul:
        agent_outputs.action['step_mul'] = agent_outputs.step_mul
        agent_outputs.action_logits['step_mul'] = agent_outputs.step_mul_logits
        learner_outputs.action_logits['step_mul'] = learner_outputs.step_mul_logits
        agent_outputs_from_buffer.action_logits['step_mul'] = agent_outputs_from_buffer.step_mul_logits
        learner_outputs_from_buffer.action_logits['step_mul'] = learner_outputs_from_buffer.step_mul_logits

    actions = tf.nest.flatten(tf.nest.map_structure(lambda x: tf.squeeze(x, axis=2), agent_outputs.action))
    behaviour_logits = tf.nest.flatten(agent_outputs.action_logits)
    target_logits = tf.nest.flatten(learner_outputs.action_logits)
    behaviour_logits_from_buffer = tf.nest.flatten(agent_outputs_from_buffer.action_logits)
    target_logits_from_buffer = tf.nest.flatten(learner_outputs_from_buffer.action_logits)

    behaviour_neg_log_probs = sum(tf.nest.map_structure(vtrace.log_probs_from_logits_and_actions, behaviour_logits, actions))
    target_neg_log_probs = sum(tf.nest.map_structure(vtrace.log_probs_from_logits_and_actions, target_logits, actions))
    entropy = sum(tf.nest.map_structure(compute_entropy_loss, target_logits))

    with tf.device('/cpu'):
        vtrace_returns = vtrace.from_importance_weights(
            log_rhos=behaviour_neg_log_probs - target_neg_log_probs,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value
        )

    advantages = tf.stop_gradient(vtrace_returns.pg_advantages)
    # Clip advantages to strictly positive:
    if clip_advantage:
        advantages *= tf.where(advantages > 0.0, tf.ones_like(advantages), tf.zeros_like(advantages))
    policy_gradient_loss = tf.reduce_sum(target_neg_log_probs * tf.stop_gradient(vtrace_returns.pg_advantages))
    baseline_loss = .5 * tf.reduce_sum(tf.square(vtrace_returns.vs - learner_outputs.baseline))
    entropy_loss = -tf.reduce_sum(entropy)

    # Compute the CLEAR policy cloning loss and the value cloning as described in https://arxiv.org/abs/1811.11682:
    policy_cloning_loss = sum(tf.nest.map_structure(compute_policy_cloning_loss, target_logits_from_buffer,
                                                    behaviour_logits_from_buffer))
    value_cloning_loss = tf.reduce_sum(
        tf.square(learner_outputs_from_buffer.baseline - tf.stop_gradient(agent_outputs_from_buffer.baseline)))

    # Combine individual losses, weighted by cost factors, to build overall loss:
    total_loss = policy_gradient_loss \
                 + baseline_cost * baseline_loss \
                 + entropy_cost * entropy_loss \
                 + policy_cloning_cost * policy_cloning_loss \
                 + value_cloning_cost * value_cloning_loss

    optimizer = tf.train.AdamOptimizer(learning_rate, adam_beta1, adam_beta2, adam_epsilon)
    parameters = tf.trainable_variables()
    gradients = tf.gradients(total_loss, parameters)
    gradients, grad_norm = clip_gradients(gradients, clip_grad_norm)
    train_op = optimizer.apply_gradients(list(zip(gradients, parameters)))

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        if fixed_step_mul:
            step_env_frames = unroll_length * (batch_size - batch_size_from_replay) * step_mul
        else:
            # do not use replay samples to calculate num environment frames
            step_env_frames = tf.to_int64(tf.reduce_sum(learner_outputs.step_mul[:, batch_size_from_replay:] + 1))
        num_env_frames_and_train = tf.train.get_global_step().assign_add(step_env_frames)

    # Adding a few summaries.
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('entropy_cost', entropy_cost)
    tf.summary.scalar('loss/policy_gradient', policy_gradient_loss)
    tf.summary.scalar('loss/baseline', baseline_loss)
    tf.summary.scalar('loss/entropy', entropy_loss)
    tf.summary.scalar('loss/policy_cloning', policy_cloning_loss)
    tf.summary.scalar('loss/value_cloning', value_cloning_loss)
    tf.summary.scalar('loss/total_loss', total_loss)
    for action_name, action in agent_outputs.action.items():
        tf.summary.histogram(f'action/{action_name}', action)
    tf.summary.scalar('grad_norm', grad_norm)

    return done, infos, num_env_frames_and_train
    #
    #
    #
    #
    # # Compute the policy gradient loss:
    # behaviour_action_log_probs, target_action_log_probs, cross_entropy = [], [], []
    # for behaviour_logits, target_logits, action in zip(tf.nest.flatten(agent_outputs.action_logits),
    #                                                    tf.nest.flatten(learner_outputs.action_logits),
    #                                                    tf.nest.flatten(agent_outputs.action)):
    #     action = tf.squeeze(tf.cast(action, dtype=tf.int32), axis=2)
    #     behaviour_action_log_probs.append(vtrace.log_probs_from_logits_and_actions(behaviour_logits, action))
    #     target_action_log_probs.append(vtrace.log_probs_from_logits_and_actions(target_logits, action))
    #     cross_entropy.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=target_logits, labels=action))
    #
    # with tf.device('/cpu'):
    #     vtrace_returns = vtrace.from_log_probs(
    #         behaviour_action_log_probs=sum(behaviour_action_log_probs),
    #         target_action_log_probs=sum(target_action_log_probs),
    #         discounts=discounts,
    #         rewards=clipped_rewards,
    #         values=learner_outputs.baseline,
    #         bootstrap_value=bootstrap_value
    #     )
    #
    # advantages = tf.stop_gradient(vtrace_returns.pg_advantages)
    #
    # # Clip advantages to strictly positive
    # if clip_advantage:
    #     advantages *= tf.where(advantages > 0.0, tf.ones_like(advantages), tf.zeros_like(advantages))
    #
    # policy_gradient_loss = tf.reduce_sum(sum(cross_entropy) * advantages)
    #
    # # Compute the baseline loss:
    # baseline_loss = compute_baseline_loss(vtrace_returns.vs - learner_outputs.baseline)
    #
    # # Compute the entropy regularization loss:
    # entropy_loss = -tf.reduce_sum(sum([
    #     compute_entropy_loss(target_logits) for target_logits in tf.nest.flatten(learner_outputs.action_logits)
    # ]))
    #
    # # Compute the policy cloning loss and the value cloning as described in
    # # `Experience Replay for Continual Learning` (https://arxiv.org/abs/1811.11682):
    # policy_cloning_loss = sum([
    #     compute_policy_cloning_loss(replay_target_logits, replay_behaviour_logits)
    #     for replay_behaviour_logits, replay_target_logits
    #     in zip(tf.nest.flatten(agent_outputs_from_buffer.action_logits),
    #            tf.nest.flatten(learner_outputs_from_buffer.action_logits))
    # ])
    # value_cloning_loss = tf.reduce_sum(
    #     tf.square(learner_outputs_from_buffer.baseline - tf.stop_gradient(agent_outputs_from_buffer.baseline)))
    #
    #
    #
    #
    #
    # # Combine individual losses, weighted by cost factors, to build overall loss:
    # total_loss = policy_gradient_loss \
    #              + baseline_cost * baseline_loss \
    #              + entropy_cost * entropy_loss \
    #              + policy_cloning_cost * policy_cloning_loss \
    #              + value_cloning_cost * value_cloning_loss
    #
    # # Optimization
    # optimizer = tf.train.AdamOptimizer(learning_rate, adam_beta1, adam_beta2, adam_epsilon)
    # parameters = tf.trainable_variables()
    # gradients = tf.gradients(total_loss, parameters)
    # gradients, grad_norm = clip_gradients(gradients, clip_grad_norm)
    # train_op = optimizer.apply_gradients(list(zip(gradients, parameters)))
    #
    # # Merge updating the network and environment frames into a single tensor.
    # with tf.control_dependencies([train_op]):
    #     if fixed_step_mul:
    #         step_env_frames = unroll_length * (batch_size - batch_size_from_replay) * step_mul
    #     else:
    #         # do not use replay samples to calculate num environment frames
    #         step_env_frames = tf.to_int64(tf.reduce_sum(learner_outputs.step_mul[:, batch_size_from_replay:] + 1))
    #     num_env_frames_and_train = tf.train.get_global_step().assign_add(step_env_frames)
    #
    # # Adding a few summaries.
    # tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.scalar('entropy_cost', entropy_cost)
    # tf.summary.scalar('loss/policy_gradient', policy_gradient_loss)
    # tf.summary.scalar('loss/baseline', baseline_loss)
    # tf.summary.scalar('loss/entropy', entropy_loss)
    # tf.summary.scalar('loss/policy_cloning', policy_cloning_loss)
    # tf.summary.scalar('loss/value_cloning', value_cloning_loss)
    # tf.summary.scalar('loss/total_loss', total_loss)
    # for action_name, action in agent_outputs.action.items():
    #     tf.summary.histogram(f'action/{action_name}', action)
    # tf.summary.scalar('grad_norm', grad_norm)
    #
    # return done, infos, num_env_frames_and_train


def build_critic_learner(agent: snt.RNNCore, agent_state, env_outputs, agent_outputs, reward_clipping: str,
                         discounting: float, clip_grad_norm: float, learning_rate: float, batch_size: int,
                         batch_size_from_replay: int, unroll_length: int, reward_scaling: float = 1.0,
                         adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8,
                         fixed_step_mul: bool = False, step_mul: int = 8):
    learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs, agent_state)

    bootstrap_value = learner_outputs.baseline[-1]
    rewards, infos, done, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
    learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

    rewards = rewards * reward_scaling
    clipped_rewards = clip_rewards(rewards, reward_clipping)
    discounts = tf.to_float(~done) * discounting

    returns = tf.scan(lambda a, x: x[0] + x[1] * a,
                      elems=[clipped_rewards, discounts],
                      initializer=bootstrap_value,
                      parallel_iterations=1,
                      reverse=True,
                      back_prop=False)

    baseline_loss = .5 * tf.reduce_sum(tf.square(returns - learner_outputs.baseline))

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate, adam_beta1, adam_beta2, adam_epsilon)
    parameters = tf.trainable_variables()
    gradients = tf.gradients(baseline_loss, parameters)
    gradients, grad_norm = clip_gradients(gradients, clip_grad_norm)
    train_op = optimizer.apply_gradients(list(zip(gradients, parameters)))

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        if fixed_step_mul:
            step_env_frames = unroll_length * (batch_size - batch_size_from_replay) * step_mul
        else:
            # do not use replay samples to calculate num environment frames
            step_env_frames = tf.to_int64(tf.reduce_sum(learner_outputs.step_mul[:, batch_size_from_replay:] + 1))
        num_env_frames_and_train = tf.train.get_global_step().assign_add(step_env_frames)

    # Adding a few summaries.
    tf.summary.scalar('ciritc_pretrain/learning_rate', learning_rate, ['ciritc_pretrain_summaries'])
    tf.summary.scalar('ciritc_pretrain/baseline_loss', baseline_loss, ['ciritc_pretrain_summaries'])
    tf.summary.scalar('ciritc_pretrain/grad_norm', grad_norm, ['ciritc_pretrain_summaries'])

    summary_op = tf.summary.merge_all('ciritc_pretrain_summaries')

    return num_env_frames_and_train, summary_op
