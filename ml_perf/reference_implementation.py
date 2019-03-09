# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

from collections import OrderedDict
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import utils

from absl import app, flags
from rl_loop import example_buffer, fsdb

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_float('overwhelming_win_rate', 0.80,
                   'Decide whether a new model win over old model with high '
                   'win rate.')

flags.DEFINE_float('low_win_rate', 0.20,
                   'Decide whether a new model win over old model with low '
                   'win rate.')

flags.DEFINE_float('bias_threshold', 0.70,
                   'Decide whether a game result is high biased.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 5,
                     'Maximum number of recent selfplay rounds to train on.')

FLAGS = flags.FLAGS


# Models are named with the current reinforcement learning loop iteration number
# and the model generation (how many models have passed gating). For example, a
# model named "000015-000007" was trained on the 15th iteration of the loop and
# is the 7th models that passed gating.
# Note that we rely on the iteration number being the first part of the model
# name so that the training chunks sort correctly.
class State:

  def __init__(self):
    self.iter_num = 0
    self.gen_num = 0

    # We start playing using a random model.
    # After the first round of selfplay has been completed, the engine is
    # updated to FLAGS.engine.
    self.engine = 'random'

    self.best_model_name = 'random'

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def seed(self):
    return self.iter_num + 1


def expand_flags(cmd, *args):
  """Expand & dedup any flagfile command line arguments."""

  # Read any flagfile arguments and expand them into a new list.
  expanded = flags.FlagValues().read_flags_from_files(args)

  # When one flagfile includes & overrides a base one, the expanded list may
  # contain multiple instances of the same flag with different values.
  # Deduplicate, always taking the last occurance of the flag.
  deduped = OrderedDict()
  for arg in expanded:
    flag = arg.split('=', 1)[0]
    deduped[flag] = arg
  return deduped.values()


def checked_run(name, *cmd):
  # Log the expanded & deduped list of command line arguments, so we can know
  # exactly what's going on. Note that we don't pass the expanded list of
  # arguments to the actual subprocess because of a quirk in how unknown flags
  # are handled: unknown flags in flagfiles are silently ignored, while unknown
  # flags on the command line will cause the subprocess to abort.
  logging.info(
      'Running %s:\n  %s  %s', name, cmd[0], '  '.join(expand_flags(*cmd)))

  with utils.logged_timer('%s finished' % name.capitalize()):
    completed_process = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed_process.returncode:
      logging.error('Error running %s: %s', name,
                    completed_process.stdout.decode())
      raise RuntimeError('Non-zero return code executing %s' % ' '.join(cmd))
  return completed_process


def get_lines(completed_process, slice):
  return '\n'.join(completed_process.stdout.decode()[:-1].split('\n')[slice])


class MakeSlice(object):

  def __getitem__(self, item):
    return item


make_slice = MakeSlice()


# Return up to num_records of golden chunks to train on.
def get_golden_chunk_records(num_records):
  # Sort the list of chunks so that the most recent ones are first and return
  # the requested prefix.
  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:num_records]


# Self-play a number of games.
def selfplay(state, flagfile='selfplay'):
  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  model_path = os.path.join(fsdb.models_dir(), state.best_model_name)
  sgf_dir = os.path.join(fsdb.sgf_dir(), state.output_model_name)

  result = checked_run('selfplay',
      'bazel-bin/cc/selfplay',
      '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
      '--model={}.pb'.format(model_path),
      '--output_dir={}'.format(output_dir),
      '--holdout_dir={}'.format(holdout_dir),
      '--sgf_dir={}'.format(sgf_dir),
      '--sgf_timestamp=false',
      '--seed={}'.format(state.seed))
  logging.info(get_lines(result, make_slice[-2:]))

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*', '*.zz')
  random.seed(state.seed)
  tf.set_random_seed(state.seed)
  np.random.seed(state.seed)
  # TODO(tommadams): This method of generating one golden chunk per generation
  # is sub-optimal because each chunk gets reused multiple times for training,
  # introducing bias. Instead, a fresh dataset should be uniformly sampled out
  # of *all* games in the training window before the start of each training run.
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)

  # TODO(tommadams): parallel_fill is currently non-deterministic. Make it not
  # so.
  logging.info('Writing golden chunk from "{}"'.format(pattern))
  buffer.parallel_fill(tf.gfile.Glob(pattern))
  buffer.flush(os.path.join(fsdb.golden_chunk_dir(),
                            state.output_model_name + '.tfrecord.zz'))
  sgf_path = os.path.join(sgf_dir, "clean/*.sgf")
  sgf_files = glob.glob(sgf_path)
  full_sgf_path = os.path.join(sgf_dir, "full")
  white_win = 0
  black_win = 0
  for file_name in sgf_files:
    if 'W+' in open (file_name).read():
        white_win += 1
    if 'B+' in open (file_name).read():
        black_win += 1
  logging.info ("White win {} times, black win {} times.".format(white_win, black_win))
  bias = abs(white_win - black_win)/(white_win+black_win)
  logging.info ("selfplay bias = {}".format(bias))
  # remove full_sgf_files to save space
  with utils.logged_timer('remove sgf files'):
    logging.info('removing {}'.format(full_sgf_path))
    shutil.rmtree(full_sgf_path, ignore_errors=True)
  return bias



# Train a new model.
def train(state, tf_records):
  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  checked_run('training',
      'python3', 'train.py', *tf_records,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--training_seed={}'.format(state.seed),
      '--freeze=true')


# Validate the trained model against holdout games.
def validate(state, holdout_glob):
  checked_run('validation',
      'python3', 'validate.py', holdout_glob,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()))


# Evaluate the trained model against the current best model.
def evaluate(state):
  eval_model = state.train_model_name
  best_model = state.best_model_name
  eval_model_path = os.path.join(fsdb.models_dir(), eval_model)
  best_model_path = os.path.join(fsdb.models_dir(), best_model)
  sgf_dir = os.path.join(fsdb.eval_dir(), eval_model)
  result = checked_run('evaluation',
      'bazel-bin/cc/eval',
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'eval.flags')),
      '--model={}.pb'.format(eval_model_path),
      '--model_two={}.pb'.format(best_model_path),
      '--sgf_dir={}'.format(sgf_dir),
      '--seed={}'.format(state.seed))
  result = get_lines(result, make_slice[-7:])
  logging.info(result)
  pattern = '{}\s+(\d+)\s+\d+\.\d+%\s+(\d+)\s+\d+\.\d+%\s+(\d+)'.format(
            eval_model)
  matches = re.findall(pattern, result)
  total = 0
  for i in range(len(matches)):
    total += int(matches[i][0])
  # TODO needs to use actual number of names in flags to replace 100.0 below
  win_rate = total / 100.0
  logging.info('Win rate %s vs %s: %.3f', eval_model, best_model, win_rate)
  return win_rate


def rl_loop():
  state = State()

  # Play the first round of selfplay games with a fake model that returns
  # random noise. We do this instead of playing multiple games using a single
  # model bootstrapped with random noise to avoid any initial bias.
  selfplay(state, 'bootstrap')

  # Train a real model from the random selfplay games.
  tf_records = get_golden_chunk_records(1)
  state.iter_num += 1
  train(state, tf_records)

  # Select the newly trained model as the best.
  state.best_model_name = state.train_model_name
  state.gen_num += 1

  # Run selfplay using the new model.
  bias = selfplay(state)

  # Rounds we considered generated by a competitive player
  # If a set of game is played by a non-competitive player, these games
  # won't be used to train a strong model
  competitive_iter_count = FLAGS.window_size

  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    # Build holdout glob before incrementing the iteration number because we
    # want to run validation on the previous generation.
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')

    # Train on shuffled game data from recent selfplay rounds, ignoring the
    # random bootstrapping round.
    # TODO(tommadams): potential improvments:
    #   - "slow window": increment number of models in window by 1 every 2
    #     generations.
    #   - uniformly resample the window each iteration (see TODO in selfplay
    #     for more info).
    tf_records = get_golden_chunk_records(min(min(FLAGS.window_size,
                                                  state.iter_num),
                                                  competitive_iter_count))
    state.iter_num += 1
    train(state, tf_records)

    # These could all run in parallel.
    validate(state, holdout_glob)
    model_win_rate = evaluate(state)

    # TODO(tommadams): if a model doesn't get promoted after N iterations,
    # consider deleting the most recent N training checkpoints because training
    # might have got stuck in a local minima.
    if model_win_rate >= FLAGS.gating_win_rate:
      # Promote the trained model to the best model and increment the generation
      # number.
      # Tentatively promote current model and run a round of selfplay
      temp_best_model_name = state.best_model_name
      state.best_model_name = state.train_model_name
      state.gen_num += 1
      bias = selfplay(state)
      if bias > FLAGS.bias_threshold:
        # Giveup promoting this model because new model is a biased model
        state.best_model_name = temp_best_model_name
        state.gen_num -= 1
        # Regenerate selfplay data using previous model
        tf_records = get_golden_chunk_records(1)
        logging.info('Burying {} for selfplay bias > {}.'.format(tf_records[0],
                     FLAGS.bias_threshold))
        shutil.move(tf_records[0], tf_records[0] + '.bury')
        bias = selfplay(state)
      elif model_win_rate >= FLAGS.overwhelming_win_rate:
        # in the case that the promoted model win overwhelmingly over the old
        # model, consider the old model non-competitive.  We re-do selfplay
        # with the new best model and start training from new game plays
        competitive_iter_count = 1
    else:
      bias = selfplay(state)
    competitive_iter_count += 1


def main(unused_argv):
  """Run the reinforcement learning loop."""

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)

  utils.ensure_dir_exists(fsdb.models_dir())
  utils.ensure_dir_exists(fsdb.selfplay_dir())
  utils.ensure_dir_exists(fsdb.holdout_dir())
  utils.ensure_dir_exists(fsdb.eval_dir())
  utils.ensure_dir_exists(fsdb.golden_chunk_dir())
  utils.ensure_dir_exists(fsdb.working_dir())

  # Copy the target model to the models directory so we can find it easily.
  shutil.copy('ml_perf/target.pb', fsdb.models_dir())

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'reinforcement.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  with utils.logged_timer('Total time'):
    rl_loop()


if __name__ == '__main__':
  app.run(main)
