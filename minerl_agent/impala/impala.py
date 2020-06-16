import collections
import contextlib
import multiprocessing
import os
import sys
import traceback
from typing import Callable

import numpy as np
import tensorflow as tf

from minerl_agent.environment.actions import ActionSpace
from minerl_agent.environment.observations import ObservationSpace
from minerl_agent.impala import py_process, learner
from minerl_agent.impala.buffer import Buffer
from minerl_agent.impala.environments import PyProcessMineRLObtain, FlowEnvironment

try:
    import minerl_agent.impala.dynamic_batching
except tf.errors.NotFoundError:
    tf.logging.warning('Running without dynamic batching.')

from six.moves import range

nest = tf.nest

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple('ActorOutput', 'level_name agent_state env_outputs agent_outputs')


def is_single_machine(task_id):
    return task_id == -1


def build_actor(agent, env, level_name, unroll_length: int, observation_space: ObservationSpace):
    """Builds the actor loop."""
    # Initial values.
    initial_env_output, initial_env_state = env.initial()
    initial_agent_state = agent.initial_state(1)
    initial_action = nest.map_structure(lambda _: tf.zeros([1], dtype=tf.int32), agent.output_size[0])
    dummy_agent_output, _ = agent(
        (initial_action,
         nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
        initial_agent_state
    )
    initial_agent_output = nest.map_structure(
        lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

    # All state that needs to persist across training iterations. This includes
    # the last environment output, agent state and last agent output. These
    # variables should never go on the parameter servers.
    def create_state(t):
        # Creates a unique variable scope to ensure the variable name is unique.
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state, (initial_env_state, initial_env_output, initial_agent_state,
                       initial_agent_output))

    def step(input_, unused_i):
        """Steps through the agent and the environment."""
        env_state, env_output, agent_state, agent_output = input_

        # Run agent.
        action = agent_output[0]
        batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                env_output)
        agent_output, agent_state = agent((action, batched_env_output), agent_state)

        # Convert action index to the native action.
        # action = agent_output[0][0]
        # raw_action = tf.gather(action_set, action)

        env_output, env_state = env.step(agent_output[0], agent_output[2], env_state)

        # env_output = nest.pack_sequence_as(observation_specs, env_output)

        return env_state, env_output, agent_state, agent_output

    # Run the unroll. `read_value()` is needed to make sure later usage will
    # return the first values and not a new snapshot of the variables.
    first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
    _, first_env_output, first_agent_state, first_agent_output = first_values

    # Use scan to apply `step` multiple times, therefore unrolling the agent
    # and environment interaction for `unroll_length`. `tf.scan` forwards
    # the output of each call of `step` as input of the subsequent call of `step`.
    # The unroll sequence is initialized with the agent and environment states
    # and outputs as stored at the end of the previous unroll.
    # `output` stores lists of all states and outputs stacked along the entire
    # unroll. Note that the initial states and outputs (fed through `initializer`)
    # are not in `output` and will need to be added manually later.
    output = tf.scan(step, tf.range(unroll_length), first_values)
    _, env_outputs, _, agent_outputs = output

    # Update persistent state with the last output from the loop.
    assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                    persistent_state, output)

    # The control dependency ensures that the final agent and environment states
    # and outputs are stored in `persistent_state` (to initialize next unroll).
    with tf.control_dependencies(nest.flatten(assign_ops)):
        # Remove the batch dimension from the agent state/output.
        first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
        first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
        agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

        # Concatenate first output and the unroll along the time dimension.
        full_agent_outputs, full_env_outputs = nest.map_structure(
            lambda first, rest: tf.concat([[first], rest], 0),
            (first_agent_output, first_env_output), (agent_outputs, env_outputs))

        output = ActorOutput(
            level_name=level_name, agent_state=first_agent_state,
            env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

        # No backpropagation should be done here.
        return nest.map_structure(tf.stop_gradient, output)


def create_environment(env_config, seed, env_id, is_test=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""
    p = py_process.PyProcess(PyProcessMineRLObtain, env_config, seed)
    return FlowEnvironment(p.proxy, env_id)


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs

def train(logdir, loaddir, environment_name, action_space: ActionSpace, observation_space: ObservationSpace,
          agent_fn: Callable, num_actors: int, batch_size: int, unroll_length: int, reward_clipping: str,
          discounting: float, total_environment_frames: int, baseline_cost: float, entropy_cost: float,
          policy_cloning_cost: float, value_cloning_cost: float,
          clip_grad_norm: float, clip_advantage: bool, learning_rate: float, replay_buffer_size: int,
          fixed_step_mul: bool, num_critic_pretrain_frames: int,
          max_episode_length: int,
          replay_proportion: float, reward_scaling: float,
          step_mul: int = 8, task: int = -1, job_name: str = 'learner', seed: int = 0,
          malmo_base_port: int = None):
    """Train."""
    manager = multiprocessing.Manager()
    launch_lock = manager.Lock()
    num_env_frames = manager.Value('i', 0)
    max_episode_length = manager.Value('i', max_episode_length)
    num_env_frames_lock = manager.Lock()

    env_config = {
        'launch_lock': launch_lock,
        'action_space': action_space,
        'observation_space': observation_space,
        'fixed_step_mul': fixed_step_mul,
        'num_env_frames': num_env_frames,
        'num_env_frames_lock': num_env_frames_lock,
        'step_mul': step_mul,
        'environment_name': environment_name,
        'malmo_base_port': malmo_base_port,
        'max_episode_length': max_episode_length
    }

    if is_single_machine(task):
        local_job_device = ''
        shared_job_device = ''
        is_actor_fn = lambda i: True
        is_learner = True
        global_variable_device = '/gpu'
        filters = []
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        config.gpu_options.allow_growth = True
        server = tf.train.Server.create_local_server(config=config)
    else:
        local_job_device = '/job:%s/task:%d' % (job_name, task)
        shared_job_device = '/job:learner/task:0'
        is_actor_fn = lambda i: job_name == 'actor' and i == task
        is_learner = job_name == 'learner'

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        global_variable_device = shared_job_device + '/cpu'
        cluster = tf.train.ClusterSpec({
            'actor': ['localhost:%d' % (8001 + i) for i in range(num_actors)],
            'learner': ['localhost:8000']
        })
        filters = [shared_job_device, local_job_device]
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster, job_name=job_name, task_index=task, config=config)

    # Only used to find the actor output structure.
    with tf.Graph().as_default():
        agent = agent_fn()
        env = create_environment(env_config, env_id=1, seed=1)
        structure = build_actor(agent, env, environment_name, unroll_length, observation_space)
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), \
         tf.device(local_job_device + '/cpu'), \
         pin_global_variables(global_variable_device):
        tf.set_random_seed(seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
            agent = agent_fn()

            if is_single_machine(task) and 'dynamic_batching' in sys.modules:
                # For single machine training, we use dynamic batching for improved GPU
                # utilization. The semantics of single machine training are slightly
                # different from the distributed setting because within a single unroll
                # of an environment, the actions may be computed using different weights
                # if an update happens within the unroll.
                old_build = agent._build

                @dynamic_batching.batch_fn
                def build(*args):
                    with tf.device('/gpu'):
                        return old_build(*args)

                tf.logging.info('Using dynamic batching.')
                agent._build = build

        # Build actors and ops to enqueue their output.
        enqueue_ops = []
        for i in range(num_actors):
            if is_actor_fn(i):
                level_name = environment_name
                tf.logging.info('Creating actor %d with level %s', i, level_name)
                env = create_environment(env_config, env_id=i + 2, seed=i + 1)
                actor_output = build_actor(agent, env, level_name, unroll_length, observation_space)
                with tf.device(shared_job_device):
                    enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_learner and enqueue_ops:
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            tf.get_variable(
                'num_env_frames',
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Create batch (time major) and recreate structure.
            if replay_buffer_size > 0:
                batch_size_from_replay = int(batch_size * replay_proportion)
                batch_size_from_actors = batch_size - batch_size_from_replay
                replay_buffer = Buffer(size=replay_buffer_size, batch_size_in=batch_size_from_actors,
                                       batch_size_out=batch_size_from_replay, unroll_length=unroll_length,
                                       shapes=shapes, dtypes=dtypes)

                def sample_and_put_replay_buffer(*trajectories):
                    replay_samples, indices = replay_buffer.get()
                    replay_buffer.put(trajectories)
                    return replay_samples + [indices.astype(np.int32)]

                def put_replay_buffer(*trajectories):
                    replay_buffer.put(trajectories)
                    return np.dtype('int32').type(len(replay_buffer))

                dequeued = queue.dequeue_many(batch_size_from_actors)

                dequeued_replay = tf.py_func(sample_and_put_replay_buffer, dequeued, dtypes + [tf.int32])
                dequeued_replay, dequeued_indices = dequeued_replay[:-1], dequeued_replay[-1]
                tf.summary.histogram('experience_buffer/dequeued_indices', dequeued_indices)
                fill_replay_buffer = tf.py_func(put_replay_buffer, dequeued, tf.int32)

                # Merge data from actor with data from experience replay (50-50)
                dequeued = nest.map_structure(lambda x, y: tf.concat([x, y], axis=0), dequeued_replay, dequeued)
                dequeued = nest.pack_sequence_as(structure, dequeued)
            else:
                batch_size_from_replay = 0
                dequeued = queue.dequeue_many(batch_size)
                dequeued = nest.pack_sequence_as(structure, dequeued)

            def make_time_major(s):
                return nest.map_structure(
                    lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

            dequeued = dequeued._replace(
                env_outputs=make_time_major(dequeued.env_outputs),
                agent_outputs=make_time_major(dequeued.agent_outputs))

            with tf.device('/gpu'):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(dequeued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output])
                stage_op = area.put(flattened_output)

                data_from_actors = nest.pack_sequence_as(structure, area.get())

                if num_critic_pretrain_frames > 0:
                    # Unroll agent on sequence, create losses and update ops.
                    output_critic_pretrain = learner.build_critic_learner(agent, data_from_actors.agent_state,
                                                                          data_from_actors.env_outputs,
                                                                          data_from_actors.agent_outputs,
                                                                          reward_clipping,
                                                                          discounting, clip_grad_norm,
                                                                          learning_rate,
                                                                          batch_size, batch_size_from_replay,
                                                                          unroll_length,
                                                                          reward_scaling=reward_scaling,
                                                                          fixed_step_mul=fixed_step_mul,
                                                                          step_mul=step_mul)

                if replay_buffer_size > 0:
                    num_env_frames_placeholder = tf.placeholder(tf.int64, shape=())
                else:
                    num_env_frames_placeholder = tf.train.get_global_step()

                # Unroll agent on sequence, create losses and update ops.
                learning_rate = tf.train.polynomial_decay(learning_rate, num_env_frames_placeholder,
                                                          total_environment_frames, 0)
                output = learner.build_learner(agent, data_from_actors.agent_state, data_from_actors.env_outputs,
                                               data_from_actors.agent_outputs, reward_clipping, discounting,
                                               baseline_cost, entropy_cost, policy_cloning_cost,
                                               value_cloning_cost, clip_grad_norm,
                                               clip_advantage, learning_rate, batch_size, batch_size_from_replay,
                                               unroll_length,
                                               reward_scaling=reward_scaling, fixed_step_mul=fixed_step_mul,
                                               step_mul=step_mul)

        scaffold = None

        if is_learner and loaddir is not None and loaddir != 'None':
            if not os.path.isfile(os.path.join(loaddir, 'checkpoint')):
                raise ValueError(f"No checkpoint found in loaddir '{loaddir}'")

            # variables_to_restore = tf.contrib.framework.get_variables_to_restore()
            init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                tf.train.latest_checkpoint(loaddir),
                tf.trainable_variables(),
                ignore_missing_vars=True
            )
            scaffold = tf.train.Scaffold(
                saver=tf.train.Saver(),
                init_fn=lambda _, sess: sess.run(init_assign_op, init_feed_dict)
            )

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
        try:
            with tf.train.MonitoredTrainingSession(
                    server.target,
                    is_chief=is_learner,
                    checkpoint_dir=logdir,
                    save_checkpoint_secs=600,
                    save_summaries_secs=30,
                    log_step_count_steps=50000,
                    # config=config,
                    hooks=[py_process.PyProcessHook()],
                    scaffold=scaffold
            ) as session:
                if is_learner:
                    # Logging.
                    summary_writer = tf.summary.FileWriterCache.get(logdir)

                    if replay_buffer_size > 0:
                        # Fill replay buffer
                        tf.logging.info('Fill experience replay buffer ...')
                        current_replay_buffer_size = 0
                        while current_replay_buffer_size < replay_buffer_size:
                            current_replay_buffer_size = session.run_step_fn(
                                lambda step_context: step_context.session.run(fill_replay_buffer))
                            tf.logging.info(
                                f'Fill experience replay buffer ... {current_replay_buffer_size}/{replay_buffer_size}')
                        tf.logging.info(f"Experience replay buffer filled!")
                        tf.logging.info(f"Total environemnt frames: {num_env_frames.value}/{total_environment_frames}")

                    if num_critic_pretrain_frames > 0:
                        tf.logging.info(f'Start critic pretraining for {num_critic_pretrain_frames} frames ...')

                        # keep track of frame count at start of critic pretraining
                        start_critic_pretrain_env_frames = num_env_frames.value

                        # Execute pretraining of critic
                        while num_env_frames.value < total_environment_frames and \
                                num_env_frames.value - start_critic_pretrain_env_frames < num_critic_pretrain_frames:
                            session.run_step_fn(
                                lambda step_context: step_context.session.run(stage_op))
                            _, critic_summary = session.run_step_fn(
                                lambda step_context: step_context.session.run(output_critic_pretrain))
                            summary_writer.add_summary(critic_summary, num_env_frames.value)

                        tf.logging.info(f'Critic pretraining completed!')
                        tf.logging.info(f"Total environemnt frames: {num_env_frames.value}/{total_environment_frames}")

                    # Prepare data for first run.
                    session.run_step_fn(lambda step_context: step_context.session.run(stage_op))

                    # keep track of frame count at start of rl training
                    start_rl_env_frames = num_env_frames.value

                    # Execute learning and track performance.
                    while num_env_frames.value < total_environment_frames:
                        current_env_frames = num_env_frames.value
                        current_rl_env_frames = current_env_frames - start_rl_env_frames

                        # we only need to feed when we use a replay buffer
                        feed_dict = None
                        if replay_buffer_size > 0:
                            feed_dict = {num_env_frames_placeholder: current_env_frames}

                        level_names_v, done_v, infos_v, _, _ = session.run(
                            (data_from_actors.level_name,) + output + (stage_op,),
                            feed_dict=feed_dict
                        )

                        level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

                        for level_name, episode_return, episode_step in zip(
                                level_names_v[done_v],
                                infos_v.episode_return[done_v],
                                infos_v.episode_step[done_v]):
                            tf.logging.info('Level: %s Episode return: %f', level_name, episode_return)
                            tf.logging.info(
                                f"Total environemnt frames: {current_env_frames}/{total_environment_frames}")

                            summary = tf.summary.Summary()
                            summary.value.add(tag='episode_return', simple_value=episode_return)
                            summary_writer.add_summary(summary, current_rl_env_frames)
                    tf.logging.info(f"Impala training finished with: {num_env_frames.value}/{total_environment_frames}")
                else:
                    # Execute actors (they just need to enqueue their output).
                    while True:
                        session.run(enqueue_ops)
        except Exception as e:
            if num_env_frames.value >= total_environment_frames:
                tf.logging.info(f"Ignore exception since training finished: {e}")
            else:
                traceback.print_exc()
                raise e
