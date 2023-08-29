import os
import numpy as np
import random
import time
import wandb
import argparse
import sys
import torch
import signal
from os.path import dirname, abspath
from envs import REGISTRY as env_REGISTRY
from baselines import REGISTRY as agent_REGISTRY
from runner import REGISTRY as runner_REGISTRY
from modules.multi_processing import MultiPeocessRunner
from modules.multi_processing_double import MultiPeocessRunnerDouble
from configs.utils import get_config, recursive_dict_update, signal_handler, merge_dict


def main(args):

    default_config = get_config('experiment')
    env_config = get_config(args.env, 'envs')
    agent_config = get_config(args.agent, 'agents')



    if args.seed == None:
        args.seed = np.random.randint(0, 10000)


    if args.agent == 'tiecomm':
        args.block = 'no'
    elif args.agent == 'tiecomm_wo_inter':
        args.block = 'inter'
    elif args.agent == 'tiecomm_wo_intra':
        args.block = 'intra'


    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_config['seed'] = args.seed

    #update configs
    exp_config = recursive_dict_update(default_config,vars(args))
    env_config = recursive_dict_update(env_config, vars(args))
    agent_config = recursive_dict_update(agent_config, vars(args))


    #merge config
    config = {}
    config.update(default_config)
    config.update(env_config)
    config.update(agent_config)





    #======================================load config==============================================
    args = argparse.Namespace(**config)
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"


    #======================================wandb==============================================
    results_path = os.path.join(dirname(abspath(__file__)), "results")
    args.exp_id = f"{args.env}_{args.map}_{args.agent}_{args.memo}" #_{datetime.datetime.now().strftime('%d_%H_%M')}"

    if args.use_offline_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    tags = ['Ming', args.env, args.map, args.agent, args.memo]

    wandb.init(project='JAMMAS', name=args.exp_id, tags=tags, dir=results_path)
    wandb.config.update(args)


    #======================================register environment==============================================
    env = env_REGISTRY[args.env](env_config)

    env_info = env.get_env_info()
    agent_config['obs_shape'] = env_info["obs_shape"]
    agent_config['n_actions'] = env_info["n_actions"]
    agent_config['n_agents'] = env_info["n_agents"]
    exp_config['episode_length'] = env_info["episode_length"]
    exp_config['n_agents'] = env_info["n_agents"]
    exp_config['n_actions'] = env_info["n_actions"]

    agent = agent_REGISTRY[args.agent](agent_config)
    if args.agent=='ic3net':
        exp_config['hard_attn']=True
        exp_config['commnet']=True
        exp_config['detach_gap'] = 10
        exp_config['comm_action_one'] = True
    elif args.agent=='commnet':
        exp_config['hard_attn']=False
        exp_config['commnet']=True
        exp_config['detach_gap'] = 10
    elif args.agent=='tarmac':
        exp_config['hard_attn']=False
        exp_config['commnet']=True
        exp_config['detach_gap'] = 10
    elif args.agent=='magic':
        exp_config['hard_attn']=False
        exp_config['hid_size']=64
        exp_config['detach_gap'] = 10

    elif args.agent=='ac_att':
        exp_config['att_head']=1
        exp_config['hid_size']=64
    # elif args.agent in ['tiecomm','tiecomm_g','tiecomm_random','tiecomm_default']:
    #     exp_config['interval']= agent_config['group_interval']
    else:
        pass


    #wandb.watch(agent)

    epoch_size = exp_config['epoch_size']
    batch_size = exp_config['batch_size']
    run = runner_REGISTRY[args.agent]
    if args.use_multiprocessing:
        for p in agent.parameters():
            p.data.share_memory_()
        runner = MultiPeocessRunner(exp_config, lambda: run(exp_config, env, agent))
    else:
        runner = run(exp_config, env, agent)


    total_num_episodes = 0
    total_num_steps = 0

    for epoch in range(1, args.total_epoches+1):
        epoch_begin_time = time.time()

        log = {}
        for i in range(epoch_size):
            batch_log = runner.train_batch(batch_size)
            merge_dict(batch_log, log)
            #print(i,batch_log['success'])

        total_num_episodes += log['num_episodes']
        total_num_steps += log['num_steps']

        #print('episode_return',(log['episode_return']/log['num_episodes']))

        epoch_time = time.time() - epoch_begin_time
        wandb.log({'epoch': epoch,
                   'episode': total_num_episodes,
                   'epoch_time': epoch_time,
                   'total_steps': total_num_steps,
                   'episode_return': log['episode_return']/log['num_episodes'],
                   "episode_steps": np.mean(log['episode_steps']),
                   'action_loss': log['action_loss'],
                   'value_loss': log['value_loss'],
                   'total_loss': log['total_loss'],
                   })

        if args.agent in ['tiecomm', 'tiecomm_wo_inter', 'tiecomm_wo_intra']:
            wandb.log({'epoch': epoch,
                    #'episode': total_num_episodes,
                    'god_action_loss': log['god_action_loss'],
                    'god_value_loss': log['god_value_loss'],
                    'god_total_loss': log['god_total_loss'],
                    'num_groups': log['num_groups']/log['num_episodes'],
                    })

        if args.agent =='tiecomm_default':
            wandb.log({'epoch': epoch,
                    'num_groups': log['num_groups']/log['num_episodes'],
                    })


        if args.env == 'lbf':
            wandb.log({'epoch': epoch,
                       'episode': total_num_episodes,
                       'num_collisions':log['num_collisions']/log['num_episodes'],
                       })


        print('current epoch: {}/{}'.format(epoch, args.total_epoches))



    if sys.flags.interactive == 0 and args.use_multiprocessing:
        runner.quit()

    print("=====Done!!!=====")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TieComm')
    parser.add_argument('--memo', type=str, default="JAMMAS", help='memo name')
    parser.add_argument('--env', type=str, default="mpe", help='environment name',
                        choices=['mpe','lbf','rware','tj'])
    parser.add_argument('--map', type=str, default="mpe-large-spread-v1", help='environment map name',
                        choices=['easy','medium','hard','mpe-large-spread-v2','mpe-large-spread-v1','Foraging-easy-v0','Foraging-medium-v0','Foraging-hard-v0'])
    parser.add_argument('--time_limit', type=int, default=50, help='time limit')
    parser.add_argument('--agent', type=str, default="tiecomm_default", help='algorithm name',
                        choices=['tiecomm','tiecomm_wo_inter','tiecomm_wo_intra','tiecomm_default','ac_att','ac_mlp','gnn','commnet','ic3net','tarmac','magic'])
    # parser.add_argument('--block', type=str, default='no',choices=['no','inter','intra'], help='only works for tiecomm')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--use_offline_wandb', action='store_true', help='use offline wandb')
    parser.add_argument('--use_multiprocessing', action='store_true', help='use multiprocessing')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--total_epoches', type=int, default=1500, help='total number of training epochs')
    parser.add_argument('--n_processes', type=int, default=6, help='number of processes')
    args = parser.parse_args()

    training_begin_time = time.time()
    signal.signal(signal.SIGINT, signal_handler)
    main(args)
    training_time = time.time() - training_begin_time
    print('training time: {} h'.format(training_time/3600))
