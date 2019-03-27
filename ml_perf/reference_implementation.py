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

import asyncio
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import time
import utils
import multiprocessing
import fcntl
import glob
import threading
import copy
import time

from absl import app, flags
from rl_loop import example_buffer, fsdb
from tensorflow import gfile

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_float('overwhelming_win_rate', 1.01, #0.80,
                   'Decide whether a new model win over old model with high '
                   'win rate.')

flags.DEFINE_float('low_win_rate', -0.01, #0.20,
                   'Decide whether a new model win over old model with low '
                   'win rate.')

flags.DEFINE_float('bias_threshold', 1.01, #0.70,
                   'Decide whether a game result is high biased.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_string('check_point', True,
                    'Whether to start from a checkpoint')

flags.DEFINE_integer('max_window_size', 5,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_integer('slow_window_size', 5,
                     'Window size after which the window starts growing by '
                     '1 every slow_window_speed iterations of the RL loop.')

flags.DEFINE_integer('slow_window_speed', 1,
                     'Speed at which the training window increases in size '
                     'once the window size passes slow_window_size.')

flags.DEFINE_boolean('parallel_post_train', False,
                     'If true, run the post-training stages (eval, validation '
                     '& selfplay) in parallel.')

flags.DEFINE_string('engine', 'tf', 'The engine to use for selfplay.')

FLAGS = flags.FLAGS


class State:
  """State data used in each iteration of the RL loop.

  Models are named with the current reinforcement learning loop iteration number
  and the model generation (how many models have passed gating). For example, a
  model named "000015-000007" was trained on the 15th iteration of the loop and
  is the 7th models that passed gating.
  Note that we rely on the iteration number being the first part of the model
  name so that the training chunks sort correctly.
  """

  def __init__(self):
    self.start_time = time.time()

    self.iter_num = 0
    self.gen_num = 0

    self.best_model_name = None

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def best_model_path(self):
    if self.best_model_name is None:
      # We don't have a good model yet, use a random fake model implementation.
      return 'random:0,0.4:0.4'
    else:
      return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.best_model_name))

  @property
  def train_model_path(self):
    return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.train_model_name))

  @property
  def seed(self):
    return self.iter_num + 1


class ColorWinStats:
  """Win-rate stats for a single model & color."""

  def __init__(self, total, both_passed, opponent_resigned, move_limit_reached):
    self.total = total
    self.both_passed = both_passed
    self.opponent_resigned = opponent_resigned
    self.move_limit_reached = move_limit_reached
    # Verify that the total is correct
    assert total == both_passed + opponent_resigned + move_limit_reached


class WinStats:
  """Win-rate stats for a single model."""

  def __init__(self, line):
    pattern = '\s*(\S+)' + '\s+(\d+)' * 8
    match = re.search(pattern, line)
    if match is None:
        raise ValueError('Can\t parse line "{}"'.format(line))
    self.model_name = match.group(1)
    raw_stats = [float(x) for x in match.groups()[1:]]
    self.black_wins = ColorWinStats(*raw_stats[:4])
    self.white_wins = ColorWinStats(*raw_stats[4:])
    self.total_wins = self.black_wins.total + self.white_wins.total


def parse_win_stats_table(stats_str, num_lines):
  result = []
  lines = stats_str.split('\n')
  while True:
    # Find the start of the win stats table.
    assert len(lines) > 1
    if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
        break
    lines = lines[1:]

  # Parse the expected number of lines from the table.
  for line in lines[2:2 + num_lines]:
    result.append(WinStats(line))

  return result


def extract_multi_instance(cmd):
  cmd_list = flags.FlagValues().read_flags_from_files(cmd)
  new_cmd_list = []
  multi_instance = False
  num_instance = 0
  num_games = 0
  parallel_games = 0

  for arg in cmd_list:
    argsplit = arg.split('=', 1)
    flag = argsplit[0]
    if flag == '--multi_instance':
      if argsplit[1] == 'True':
        multi_instance = True
      else:
        multi_instance = False
    elif flag == '--num_games':
      num_games = int(argsplit[1])
    elif flag == '--parallel_games':
      parallel_games = int(argsplit[1])

  if multi_instance:
    if num_games % parallel_games != 0:
      logging.error('Error num_games must be multiply of %d', parallel_games)
      raise RuntimeError('incompatible num_games/parallel_games combination')
    num_instance = num_games//parallel_games

  for arg in cmd_list:
    argsplit = arg.split('=', 1)
    flag = argsplit[0]
    if flag == '--multi_instance':
      pass
    elif multi_instance and flag == '--num_games':
      pass
    else:
      new_cmd_list.append(arg)

  return multi_instance, num_instance, new_cmd_list

def expand_cmd_str(cmd):
  return '  '.join(flags.FlagValues().read_flags_from_files(cmd))


def get_cmd_name(cmd):
  if cmd[0] == 'python' or cmd[0] == 'python3':
    path = cmd[1]
  else:
    path = cmd[0]
  return os.path.splitext(os.path.basename(path))[0]


async def checked_run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  # Start the subprocess.
  logging.info('Running: %s', expand_cmd_str(cmd))
  with utils.logged_timer('{} finished'.format(get_cmd_name(cmd))):
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    # Stream output from the process stdout.
    chunks = []
    while True:
      chunk = await p.stdout.read(16 * 1024)
      if not chunk:
        break
      chunks.append(chunk)

    # Wait for the process to finish, check it was successful & build stdout.
    await p.wait()
    stdout = b''.join(chunks).decode()[:-1]
    if p.returncode:
      raise RuntimeError('Return code {} from process: {}\n{}'.format(
          p.returncode, expand_cmd_str(cmd), stdout))

    log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
    with gfile.Open(log_path, 'a') as f:
      f.write(expand_cmd_str(cmd))
      f.write('\n')
      f.write(stdout)
      f.write('\n')

    # Split stdout into lines.
    return stdout.split('\n')

def checked_run_mi(num_instance, *cmd):
  name = get_cmd_name(cmd)
  logging.info('Running %s*%d: %s', name, num_instance, expand_cmd_str(cmd))
  with utils.logged_timer('%s finished' % name.capitalize()):
    num_parallel_instance = int(multiprocessing.cpu_count())
    procs=[None]*num_parallel_instance
    results = [""]*num_parallel_instance
    lines = [""]*num_parallel_instance
    result_list = []

    cur_instance = 0
    # add new proc into procs
    while cur_instance < num_instance or not all (
        proc is None for proc in procs):
      if None in procs and cur_instance < num_instance:
        index = procs.index(None)
        subproc_cmd = [
                'OMP_NUM_THREADS=1',
                'KMP_AFFINITY=granularity=fine,proclist=[{}],explicit'.format(
                    ','.join(str(i) for i in list(range(
                        index, index+1)))),
                *cmd,
                '--instance_id={}'.format(cur_instance),
        ]
        subproc_cmd = ' '.join(subproc_cmd)
        if (cur_instance == 0):
          logging.info("subproc_cmd = {}".format(subproc_cmd))
        procs[index] = subprocess.Popen(subproc_cmd, shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)

        proc_count = 0
        for i in range(num_parallel_instance):
          if procs[i] != None:
            proc_count += 1
        logging.debug('started instance {} in proc {}. proc count = {}'.format(
            cur_instance, index, proc_count))

        # change stdout of the process to non-blocking
        # this is for collect output in a single thread
        flags = fcntl.fcntl(procs[index].stdout, fcntl.F_GETFL)
        fcntl.fcntl(procs[index].stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        cur_instance += 1
      for index in range(num_parallel_instance):
        if procs[index] != None:
          # collect proc output
          while True:
            try:
              line = procs[index].stdout.readline()
              if line == b'':
                break
              results[index] = results[index] + line.decode()
            except IOError:
              break

          ret_val = procs[index].poll()
          if ret_val == None:
            continue
          elif ret_val != 0:
            logging.debug(results[index])
            raise RuntimeError(
              'Non-zero return code (%d) executing %s' % (
                  ret_val, subproc_cmd))

          if index == 0:
            logging.debug(results[index])
          result_list.append(results[index])
          results[index] = ""
          procs[index] = None

          proc_count = 0
          for i in range(num_parallel_instance):
            if procs[i] != None:
              proc_count += 1
          logging.debug('proc {} finished. proc count = {}'.format(
              index, proc_count))
      time.sleep(0.001)  # avoid busy loop
    return result_list

def get_lines(completed_output, slice):
  return '\n'.join(completed_output.decode()[:-1].split('\n')[slice])

def wait(aws):
  """Waits for all of the awaitable objects (e.g. coroutines) in aws to finish.

  All the awaitable objects are waited for, even if one of them raises an
  exception. When one or more awaitable raises an exception, the exception from
  the awaitable with the lowest index in the aws list will be reraised.

  Args:
    aws: a single awaitable, or list awaitables.

  Returns:
    If aws is a single awaitable, its result.
    If aws is a list of awaitables, a list containing the of each awaitable in
    the list.

  Raises:
    Exception: if any of the awaitables raises.
  """

  aws_list = aws if isinstance(aws, list) else [aws]
  results = asyncio.get_event_loop().run_until_complete(asyncio.gather(
      *aws_list, return_exceptions=True))
  # If any of the cmds failed, re-raise the error.
  for result in results:
    if isinstance(result, Exception):
      raise result
  return results if isinstance(aws, list) else results[0]


def get_golden_chunk_records(num_records):
  """Return up to num_records of golden chunks to train on.

  Args:
    num_records: maximum number of records to return.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:num_records]


# Self-play a number of games.
async def selfplay(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """

  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)

  multi_instance, num_instance, flag_list = extract_multi_instance(
      ['--flagfile={}_mi.flags'.format(os.path.join(FLAGS.flags_dir, flagfile))])
  if not multi_instance:
    lines = await checked_run(
        'bazel-bin/cc/selfplay',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(state.best_model_path),
        '--output_dir={}'.format(output_dir),
        '--holdout_dir={}'.format(holdout_dir),
        '--seed={}'.format(state.seed))
    result = '\n'.join(lines[-6:])
    stats = parse_win_stats_table(result, 1)[0]
    logging.info(result)
    num_games = stats.total_wins
    black_total = stats.black_wins.total
    white_total = stats.white_wins.total
  else:
    result_list = checked_run_mi(
        num_instance,
        'bazel-bin/cc/selfplay',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(state.best_model_path),
        '--output_dir={}'.format(output_dir),
        '--holdout_dir={}'.format(holdout_dir),
        '--seed={}'.format(state.seed))
    num_games = 0
    black_total = 0
    white_total = 0
    for result in result_list:
      stats = parse_win_stats_table(result, 1)[0]
      num_games += stats.total_wins
      black_total += stats.black_wins.total
      white_total += stats.white_wins.total
    logging.info(result_list[0])
  logging.info('Black won %0.3f, white won %0.3f',
             black_total / num_games,
             white_total / num_games)
  bias = abs(white_total - black_total)/num_games
  logging.info('Black total %d, white total %d, total games %d, bias %0.3f.',
               black_total, white_total, num_games, bias)

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

  return bias



async def train(state, tf_records):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
    tf_records: a list of paths to TensorFlow records to train on.
  """

  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  await checked_run(
      'python3', 'train.py', *tf_records,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--training_seed={}'.format(state.seed),
      '--freeze=true')
  # Append the time elapsed from when the RL was started to when this model
  # was trained.
  elapsed = time.time() - state.start_time
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(timestamps_path, 'a') as f:
    print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)


async def validate(state, holdout_glob):
  """Validate the trained model against holdout games.

  Args:
    state: the RL loop State instance.
    holdout_glob: a glob that matches holdout games.
  """

  await checked_run(
      'python3', 'validate.py', holdout_glob,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'validate.flags')),
      '--work_dir={}'.format(fsdb.working_dir()))


async def evaluate_model(eval_model_path, target_model_path, sgf_dir, seed, flagfile='eval'):
  """Evaluate one model against a target.

  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to.
    sgf_dif: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  # TODO(tommadams): Don't append .pb to model name for random model.
  multi_instance, num_instance, flag_list = extract_multi_instance(
      ['--flagfile={}_mi.flags'.format(os.path.join(FLAGS.flags_dir, flagfile))])
  if not multi_instance:
    lines = await checked_run(
        'bazel-bin/cc/eval',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(eval_model_path),
        '--model_two={}'.format(target_model_path),
        '--sgf_dir={}'.format(sgf_dir),
        '--seed={}'.format(seed))
    result = '\n'.join(lines)
    logging.info(result)
    eval_stats, target_stats = parse_win_stats_table(result, 2)
    num_games = eval_stats.total_wins + target_stats.total_wins
    win_rate = eval_stats.total_wins / num_games
    eval_total = eval_stats.total_wins
    black_total = eval_stats.black_wins.total
    white_total = eval_stats.white_wins.total
  else:
    result_list = checked_run_mi(
        num_instance,
        'bazel-bin/cc/eval',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(eval_model_path),
        '--model_two={}'.format(target_model_path),
        '--sgf_dir={}'.format(sgf_dir),
        '--seed={}'.format(seed))
    num_games = 0
    black_total = 0
    white_total = 0
    eval_total = 0
    for result in result_list:
      eval_stats, target_stats = parse_win_stats_table(result, 2)
      num_games += eval_stats.total_wins + target_stats.total_wins
      eval_total += eval_stats.total_wins
      black_total += eval_stats.black_wins.total
      white_total += eval_stats.white_wins.total
    win_rate = eval_total / num_games
    logging.info(result_list[0])

  if eval_total != 0:
    bias = abs(white_total - black_total) / eval_total
  else:
    # by definition bias = 0.0 if eval model win zero games
    bias = 0.0
  logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
               target_stats.model_name, win_rate)
  logging.info('Black total %d, white total %d, eval total %d, bias %0.3f.',
               black_total, white_total, eval_total, bias)

  return win_rate


async def evaluate_trained_model(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return await evaluate_model(
      state.train_model_path, state.best_model_path,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed)

async def evaluate_target_model(state):
  sgf_dir = os.path.join(fsdb.eval_dir(), 'target')
  target = 'tf,' + os.path.join(fsdb.models_dir(), 'target.pb')
  return await evaluate_model(
      state.train_model_path, target, sgf_dir, state.iter_num)

def rl_loop():
  """The main reinforcement learning (RL) loop."""
  print ('Gating win rate = {}'.format(FLAGS.gating_win_rate), flush=True)

  state = State()

  if not FLAGS.check_point:
    # Play the first round of selfplay games with a fake model that returns
    # random noise. We do this instead of playing multiple games using a single
    # model bootstrapped with random noise to avoid any initial bias.
    wait(selfplay(state, 'bootstrap'))

    # Train a real model from the random selfplay games.
    tf_records = get_golden_chunk_records(1)
    state.iter_num += 1
    wait(train(state, tf_records))

    # Select the newly trained model as the best.
    state.best_model_name = state.train_model_name
    state.gen_num += 1

    # Run selfplay using the new model.
    wait(selfplay(state))
  else:
    # Rounds we considered generated by a competitive player
    # If a set of game is played by a non-competitive player, these games
    # won't be used to train a strong model
    state.best_model_name = 'start'

  competitive_iter_count = FLAGS.max_window_size

  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    # Build holdout glob before incrementing the iteration number because we
    # want to run validation on the previous generation.
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')

    # Calculate the window size from which we'll select training chunks.
    if not FLAGS.check_point:
      window = 1 + state.iter_num
    else:
      window = FLAGS.max_window_size + state.iter_num + FLAGS.slow_window_speed + 2
    if window >= FLAGS.slow_window_size:
      window = (FLAGS.slow_window_size +
                (window - FLAGS.slow_window_size) // FLAGS.slow_window_speed)
    window = min(min(window, FLAGS.max_window_size), competitive_iter_count)
    logging.info('Window size = %d', window)

    # Train on shuffled game data from recent selfplay rounds.
    tf_records = get_golden_chunk_records(window)
    state.iter_num += 1
    wait(train(state, tf_records))

    if FLAGS.parallel_post_train:
      # Run eval, validation & selfplay in parallel.
      model_win_rate, _, _ = wait([
          evaluate_trained_model(state),
          validate(state, holdout_glob),
          selfplay(state)])
    else:
      # Run eval, validation & selfplay sequentially.
      model_win_rate = wait(evaluate_trained_model(state))
      wait(validate(state, holdout_glob))
      wait(selfplay(state))

    target_win_rate = wait(evaluate_target_model(state))
    if target_win_rate >= 0.5:
      break

    # TODO(tommadams): if a model doesn't get promoted after N iterations,
    # consider deleting the most recent N training checkpoints because training
    # might have got stuck in a local minima.
    if model_win_rate >= FLAGS.gating_win_rate:
      # Promote the trained model to the best model and increment the generation
      # number.
      # Tentatively promote current model and run a round of selfplay
      #temp_best_model_name = state.best_model_name
      state.best_model_name = state.train_model_name
      state.gen_num += 1
      #bias = wait(selfplay(state))
      #if bias > FLAGS.bias_threshold:
      #  # Giveup promoting this model because new model is a biased model
      #  state.best_model_name = temp_best_model_name
      #  state.gen_num -= 1
      #  # Regenerate selfplay data using previous model
      #  tf_records = get_golden_chunk_records(1)
      #  logging.info('Burying {} for selfplay bias > {}.'.format(tf_records[0],
      #               FLAGS.bias_threshold))
      #  shutil.move(tf_records[0], tf_records[0] + '.bury')
      #  bias = wait(selfplay(state))
      #elif model_win_rate >= FLAGS.overwhelming_win_rate:
      #  # in the case that the promoted model win overwhelmingly over the old
      #  # model, consider the old model non-competitive.  We re-do selfplay
      #  # with the new best model and start training from new game plays
      #  competitive_iter_count = 1
    #else:
    #  bias = wait(selfplay(state))
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

  # Copy the flag files so there's no chance of them getting accidentally
  # overwritten while the RL loop is running.
  flags_dir = os.path.join(FLAGS.base_dir, 'flags')
  shutil.copytree(FLAGS.flags_dir, flags_dir)
  FLAGS.flags_dir = flags_dir

  # Copy the target model to the models directory so we can find it easily.
  shutil.copy('ml_perf/target.pb', fsdb.models_dir())

  if FLAGS.check_point:
    # Copy the start model to the models directory.
    shutil.copy('ml_perf/start.pb', fsdb.models_dir())

    # Copy the golden chunks to the golden chunks directory.
    shutil.copy('ml_perf/000000-000000.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000001.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000002.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000003.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000004.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000005.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000006.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000007.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000008.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000009.tfrecord.zz', fsdb.golden_chunk_dir())
    shutil.copy('ml_perf/000000-000010.tfrecord.zz', fsdb.golden_chunk_dir())

    # Copy the hold out data to the hold out directory.
    shutil.copytree('ml_perf/000000-000000',
                    os.path.join(fsdb.holdout_dir(),'000000-000000'))

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'rl_loop.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  with utils.logged_timer('Total time'):
    try:
      rl_loop()
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)
