import pathlib

from absl import flags
from multiprocessing import freeze_support

import aicrowd_helper
import datetime
import os
import sys
import traceback
import train
import test
from minerl.env import malmo

from minerl_agent.agent.agent import EmbedType, SeparateActorCriticWrapperAgent
from minerl_agent.environment.actions import CustomActionSpace
from minerl_agent.environment.observations import CustomObservationSpace, ColorSpace
from minerl_agent.agent.resnet_lstm_agent import ResnetLSTMAgent

EVALUATION_RUNNING_ON = os.getenv('EVALUATION_RUNNING_ON', None)
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'all')
EXITED_SIGNAL_PATH = os.getenv('EXITED_SIGNAL_PATH', 'shared/exited')

FLAGS = flags.FLAGS

# run params:
flags.DEFINE_string('logdir', default=f'./train/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                    help='experiment logging directory')
flags.DEFINE_string('loaddir', default=None, help='optional path to a pre-trained model')
flags.DEFINE_string('gpus', default='0', help='comma separated list of available gpus indices to use')
flags.DEFINE_integer('malmo_base_port', default=9001, help='malmo base port (to prevent conflicts)')
flags.DEFINE_integer('train_seed', default=0, help='random seed for train')
flags.DEFINE_string('test_model', default='impala', help='which model to test (bc/impala)')

# env params:
flags.DEFINE_integer('max_step_mul', default=40, help='maximum step multiplier; > 0, 1 = no step multiplier')
flags.DEFINE_integer('pov_resolution', default=32, help='pov resolution')
flags.DEFINE_enum_class('pov_color_space', default=ColorSpace.RGB, enum_class=ColorSpace, help='pov color space')
flags.DEFINE_integer('num_camera_actions', default=3, help='number of discrete camera actions (odd value!)')
flags.DEFINE_integer('camera_max_angle', default=30, help='max camera angle change in one direction')
flags.DEFINE_bool('fixed_step_mul', default=False, help='fixed step multiplier')
flags.DEFINE_integer('step_mul', default=8, help='step multiplier, if fixed_step_mul=True')

# model params:
flags.DEFINE_integer('lstm_hidden_size', default=256, help='lstm hidden size')
flags.DEFINE_boolean('use_prev_actions', default=True, help='use previous actions as input')
flags.DEFINE_enum_class('action_embed_type', default=EmbedType.EMBED, enum_class=EmbedType,
                        help='embedding method used to embed prev_actions')
flags.DEFINE_integer('action_embed_size', default=16,
                     help='size of the embedding')
flags.DEFINE_bool('separate_actor_critic', default=True, help='Separate networks for actor and critic')


# dummy flags to prevent script from failing when running with manager-mode flags
flags.DEFINE_string('seeds', default='', help='seeds')
flags.DEFINE_string('seeding_type', default='', help='seeding_type')
flags.DEFINE_string('max_instances', default='', help='max_instances')


def main():
    malmo_base_port = FLAGS.malmo_base_port
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    malmo.InstanceManager.configure_malmo_base_port(malmo_base_port)

    observation_space = CustomObservationSpace(pov_resolution=FLAGS.pov_resolution,
                                               pov_color_space=FLAGS.pov_color_space)

    action_space = CustomActionSpace(num_camera_actions=FLAGS.num_camera_actions,
                                     camera_max_angle=FLAGS.camera_max_angle)

    def combined_actor_critic_agent():
        return ResnetLSTMAgent(observation_space=observation_space,
                               action_space=action_space,
                               max_step_mul=FLAGS.max_step_mul,
                               core_hidden_size=FLAGS.lstm_hidden_size,
                               use_prev_actions=FLAGS.use_prev_actions,
                               action_embed_type=FLAGS.action_embed_type,
                               action_embed_size=FLAGS.action_embed_size)

    def separate_actor_critic_agent():
        return SeparateActorCriticWrapperAgent(actor=combined_actor_critic_agent(),
                                               critic=combined_actor_critic_agent())

    if FLAGS.separate_actor_critic:
        agent_fn = separate_actor_critic_agent
    else:
        agent_fn = combined_actor_critic_agent

    log_dir = FLAGS.logdir

    # Training Phase
    if EVALUATION_STAGE in ['all', 'training']:
        # only write out flags when training
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        FLAGS.append_flags_into_file(f'{log_dir}/flags_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.cfg')

        aicrowd_helper.training_start()
        try:
            train.main(log_dir=log_dir, load_dir=FLAGS.loaddir, observation_space=observation_space,
                       action_space=action_space, max_step_mul=FLAGS.max_step_mul, fixed_step_mul=FLAGS.fixed_step_mul,
                       step_mul=FLAGS.step_mul, agent_fn=agent_fn, seed=FLAGS.train_seed,
                       malmo_base_port=malmo_base_port)
            aicrowd_helper.training_end()
        except Exception as e:
            aicrowd_helper.training_error()
            print(traceback.format_exc())
            print(e)

    # Testing Phase
    if EVALUATION_STAGE in ['all', 'testing']:
        if EVALUATION_RUNNING_ON in ['local']:
            try:
                os.remove(EXITED_SIGNAL_PATH)
            except FileNotFoundError:
                pass
        aicrowd_helper.inference_start()
        try:
            test.main(log_dir=log_dir, test_model=FLAGS.test_model, observation_space=observation_space,
                      action_space=action_space, fixed_step_mul=FLAGS.fixed_step_mul, step_mul=FLAGS.step_mul,
                      agent_fn=agent_fn)
            aicrowd_helper.inference_end()
        except Exception as e:
            aicrowd_helper.inference_error()
            print(traceback.format_exc())
            print(e)
        if EVALUATION_RUNNING_ON in ['local']:
            from pathlib import Path
            Path(EXITED_SIGNAL_PATH).touch()

    # Launch instance manager
    if EVALUATION_STAGE in ['manager']:
        from minerl.env.malmo import launch_instance_manager
        launch_instance_manager()


if __name__ == '__main__':
    freeze_support()
    FLAGS(sys.argv)
    main()
