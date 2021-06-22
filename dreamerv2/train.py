import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
import itertools
from copy import copy

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent
import elements
import common


configs = pathlib.Path(sys.argv[0]).parent / 'configs.yaml'
configs = yaml.safe_load(configs.read_text())
config = elements.Config(configs['defaults'])
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
  config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)
logdir = pathlib.Path(config.logdir).expanduser()
config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)

tf.config.experimental_run_functions_eagerly(not config.jit)
message = 'No GPU found. To actually train on CPU remove this assert.'
assert tf.config.experimental.list_physical_devices('GPU'), message
for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)
assert config.precision in (16, 32), config.precision
if config.precision == 16:
  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

print('Logdir', logdir)
train_replay = common.Replay(logdir / 'train_replay', config.replay_size)
# eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1)
eval_grayscale_replay = common.Replay(logdir / 'eval_grayscale_replay', config.time_limit or 1)
eval_rgb_replay = common.Replay(logdir / 'eval_rgb_replay', config.time_limit or 1)
eval_jitter_inter_replay = common.Replay(logdir / 'eval_jitter_inter_replay', config.time_limit or 1)
eval_jitter_extra_replay = common.Replay(logdir / 'eval_jitter_extra_replay', config.time_limit or 1)

step = elements.Counter(train_replay.total_steps)
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    elements.TensorBoardOutput(logdir),
]
logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
metrics = collections.defaultdict(list)
should_train = elements.Every(config.train_every)
should_log = elements.Every(config.log_every)
should_video_train = elements.Every(config.eval_every)
should_video_eval = elements.Every(config.eval_every)

def make_env(mode):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = common.DMC(task, config.action_repeat, config.image_size)
    env = common.NormalizeAction(env)
  elif suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.image_size, config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    env = common.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  env = common.RewardObs(env)
  env = common.ResetObs(env)
  if config.color_jitter:
    if 'rgb' not in mode:
      print('Using color jitter augmentation...')
      env = common.ColorJitter(env, resample=config.resample_jitter)
      # return env
  if 'jitter' in mode:
    if 'inter' in mode:
      print('eval envs with interpolation jitter noise')
      env = common.ColorJitter(env, resample=config.resample_jitter)
    elif 'extra' in mode:
    # else:
      # if 'inter' in mode:
      #   print('Creating eval envs with interpolation jitter noise...')
      #   env = common.ColorJitter(env, resample=config.resample_jitter)
        # return env
      print('eval envs with extrapolation jitter noise...')
      c0_noise = [[-1,0.5],[-0.5,1]]
      c1_noise = [[0.2,1.4],[0.6,1.8]]
      env = common.ColorJitter(env,
                              brightness=c1_noise, contrast=c1_noise,
                              saturation=c1_noise, hue=c0_noise,
                              resample=config.resample_jitter)
  return env

def per_episode(ep, mode):
  # print('mode:',mode)
  length = len(ep['reward']) - 1
  score = float(ep['reward'].astype(np.float64).sum())
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  replay_ = dict(train=train_replay, # eval=eval_replay,
                 eval_grayscale=eval_grayscale_replay, eval_rgb=eval_rgb_replay,
                 eval_jitter_inter=eval_jitter_inter_replay,
            eval_jitter_extra=eval_jitter_extra_replay)[mode]
  replay_.add(ep)
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  should = {'train': should_video_train, 'eval': should_video_eval,
            'eval_grayscale':should_video_eval,
            'eval_rgb':should_video_eval, 'eval_jitter_inter':should_video_eval,
            'eval_jitter_extra':should_video_eval}[mode]
  if should(step):
    logger.video(f'{mode}_policy', ep['image'])
  logger.write()

print('Create envs.')
eval_envs_list = []
train_envs = [make_env('train') for _ in range(config.num_envs)]
if config.grayscale:
  eval_envs = [make_env('eval_grayscale') for _ in range(config.num_envs)]
  eval_envs_list.append(eval_envs)
  eval_envs_name = ['eval_grayscale']
else:
  eval_rgb_envs = [make_env('eval_rgb') for _ in range(config.num_envs)]
  eval_envs_list.append(eval_rgb_envs)
  eval_jitter_inter_envs = [make_env('eval_jitter_inter') for _ in range(config.num_envs)]
  eval_envs_list.append(eval_jitter_inter_envs)
  eval_jitter_extra_envs = [make_env('eval_jitter_extra') for _ in range(config.num_envs)]
  eval_envs_list.append(eval_jitter_extra_envs)
  eval_envs_name = ['eval_rgb','eval_jitter_inter','eval_jitter_extra']

action_space = train_envs[0].action_space['action']
train_driver = common.Driver(train_envs)
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
eval_driver_list = []
for envs, name in zip(eval_envs_list, eval_envs_name):
  # print(name)
  driver = common.Driver(envs)
  driver.on_episode(functools.partial(per_episode,mode=name))
  # driver.on_episode(lambda ep: per_episode(ep, mode=copy(name)))
  eval_driver_list.append(driver)

prefill = max(0, config.prefill - train_replay.total_steps)
if prefill:
  print(f'Prefill dataset ({prefill} steps).')
  random_agent = common.RandomAgent(action_space)
  train_driver(random_agent, steps=prefill, episodes=1)
  train_driver.reset()
  for driver in eval_driver_list:
    # print('xxxx')
    # print(driver._envs)
    driver(random_agent, episodes=1)
    driver.reset()

print('Create agent.')
train_dataset = iter(train_replay.dataset(**config.dataset))
# eval_dataset = iter(eval_replay.dataset(**config.dataset))
eval_dataset_list = []
if config.grayscale:
  dataset = iter(eval_grayscale_replay.dataset(**config.dataset))
  eval_dataset_list.append(dataset)
# eval_rgb_replay = common.Replay(logdir / 'eval_rgb_replay', config.time_limit or 1)
# eval_jitter_inter_replay = common.Replay(logdir / 'eval_jitter_inter_replay', config.time_limit or 1)
# eval_jitter_extra_replay = common.Replay(logdir / 'eval_jitter_extra_replay', config.time_limit or 1)
else:
  for replay in [eval_rgb_replay,eval_jitter_inter_replay,eval_jitter_extra_replay]:
    dataset = iter(replay.dataset(**config.dataset))
    eval_dataset_list.append(dataset)

agnt = agent.Agent(config, logger, action_space, step, train_dataset)
if (logdir / 'variables.pkl').exists():
  agnt.load(logdir / 'variables.pkl')
else:
  config.pretrain and print('Pretrain agent.')
  for _ in range(config.pretrain):
    agnt.train(next(train_dataset))

def train_step(tran):
  if should_train(step):
    for _ in range(config.train_steps):
      _, mets = agnt.train(next(train_dataset))
      [metrics[key].append(value) for key, value in mets.items()]
  if should_log(step):
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(agnt.report(next(train_dataset)), prefix='train')
    logger.write(fps=True)
train_driver.on_step(train_step)

while step < config.steps:
  logger.write()
  print('Start evaluation.')
  for eval_dataset,name in zip(eval_dataset_list,eval_envs_name):
    logger.add(agnt.report(next(eval_dataset)), prefix=name)
  eval_policy = functools.partial(agnt.policy, mode='eval')
  for driver in eval_driver_list:
    driver(eval_policy, episodes=config.eval_eps)
  print('Start training.')
  train_driver(agnt.policy, steps=config.eval_every)
  agnt.save(logdir / 'variables.pkl')
for env in train_envs + itertools.chain.from_iterable(eval_envs):
  try:
    env.close()
  except Exception:
    pass
