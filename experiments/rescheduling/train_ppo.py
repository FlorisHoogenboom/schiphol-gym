# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
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

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

import logging
import functools
import os
import time

import click
import tensorflow as tf
import tensorflow.keras as tfk
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common

from schym.gate_reassignment import GateScheduling


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_env(
    n_resources=10,
    max_len=60,
    visible_window=30,
):
    """Create the environment for training"""
    return suite_gym.wrap_env(
        GateScheduling(n_resources=n_resources, max_len=max_len, visible_window=visible_window)
    )


def get_actor_and_value_network(action_spec, observation_spec):
    preprocessing_layers = tfk.Sequential([
        tfk.layers.Lambda(lambda x: x - 0.5),  # Normalization
        tfk.layers.MaxPooling2D((5, 5), strides=(5, 5)),
        tfk.layers.Conv2D(256, (11, 3), (1, 1), padding='valid', activation='relu'),
        tfk.layers.Reshape((-1, 256)),
        tfk.layers.Conv1D(128, 1, activation='relu'),
        tfk.layers.Flatten()
    ])

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        fc_layer_params=(200, 100),
        activation_fn=tfk.activations.relu)
    value_net = value_network.ValueNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        fc_layer_params=(200, 100),
        activation_fn=tfk.activations.relu)

    return actor_net, value_net


def get_agent(
    time_step_spec,
    action_spec,
    actor_net,
    value_net,
    num_epochs,
    step_counter,
    learning_rate
):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tf_agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec,
        action_spec,
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=step_counter
    )

    return tf_agent


def get_metrics(n_parallel_env, num_eval_episodes):
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            batch_size=n_parallel_env),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=n_parallel_env),
    ]

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    return step_metrics, train_metrics, eval_metrics



@click.command()
@click.option('--root-dir')
def train_eval(
        root_dir,
        env_load_fn=get_env,
        random_seed=None,
        # Params for collect
        num_environment_steps=25000000,
        collect_episodes_per_iteration=10,
        num_parallel_environments=10,
        replay_buffer_capacity=1001,  # Per-environment
        # Params for train
        num_epochs=10,
        learning_rate=1e-4,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=500,
        # Params for summaries and logging
        train_checkpoint_interval=500,
        policy_checkpoint_interval=500,
        policy_save_interval=10000,
        log_interval=50,
        summary_interval=50,
        summaries_flush_secs=1,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False):

    if random_seed is not None:
        tf.set_random_seed(random_seed)

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

    logging.info('Running %d episodes in parallel' % num_parallel_environments)
    logging.info('Collecting %d episodes per step' % collect_episodes_per_iteration)
    logging.info('Using replay buffer capacity of %d' % replay_buffer_capacity)

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)


    eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn())
    tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn()] * num_parallel_environments
        )
    )

    actor_net, value_net = get_actor_and_value_network(tf_env.action_spec(),
                                                       tf_env.observation_spec())

    train_steps = tf.Variable(0)
    with tf.summary.record_if(lambda: tf.math.equal(train_steps % summary_interval, 0)):
        tf_agent = get_agent(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=tf_env.action_spec(),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            step_counter=train_steps,
            learning_rate=learning_rate
        )
        tf_agent.initialize()

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        step_metrics, train_metrics, eval_metrics = get_metrics(
            n_parallel_env=num_parallel_environments,
            num_eval_episodes=num_eval_episodes
        )

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity
        )

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=train_steps,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
        )
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=eval_policy,
            global_step=train_steps
        )
        saved_model = policy_saver.PolicySaver(
            eval_policy,
            train_step=train_steps
        )
        train_checkpointer.initialize_or_restore()

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_episodes_per_iteration)

        def train_step():
            trajectories = replay_buffer.gather_all()
            return tf_agent.train(experience=trajectories)

        if use_tf_functions:
            # TODO(b/123828980): Enable once the cause for slowdown was identified.
            collect_driver.run = common.function(collect_driver.run, autograph=False)
            tf_agent.train = common.function(tf_agent.train, autograph=False)
            train_step = common.function(train_step)

        collect_time = 0
        train_time = 0
        timed_at_step = global_step.numpy()

        while environment_steps_metric.result() < num_environment_steps:
            global_step_val = global_step.numpy()
            if global_step_val % eval_interval == 0:
                metric_utils.eager_compute(
                    eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )

            start_time = time.time()
            collect_driver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            replay_buffer.clear()
            train_time += time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics)

            if global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step_val, total_loss)
                steps_per_sec = (
                        (global_step_val - timed_at_step) / (collect_time + train_time))
                logging.info('%.3f steps/sec', steps_per_sec)
                logging.info('collect_time = %.3f, train_time = %.3f', collect_time,
                             train_time)
                with tf.compat.v2.summary.record_if(True):
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_save_interval == 0:
                    saved_model_path = os.path.join(
                        saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
                    saved_model.save(saved_model_path)

                timed_at_step = global_step_val
                collect_time = 0
                train_time = 0

        # One final eval before exiting.
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    train_eval(
        FLAGS.root_dir
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    multiprocessing.handle_main(functools.partial(app.run, main))
