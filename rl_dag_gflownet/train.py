import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import jax
import wandb
import os
from tqdm import trange
from numpy.random import default_rng
import pdb
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer

from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics, get_log_features
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils import io
from dag_gflownet.utils.wandb_utils import slurm_infos, table_from_dict, scatter_from_dicts
from dag_gflownet.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)


def main(args):

    wandb.init(
        project='dag-gflownet',
        group='posterior-graphs',
        tags=['gnn'],
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Create the environment
    scorer, data, graph = get_scorer(args, rng=rng)
    print(args.num_variables)
    # pdb.set_trace()
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(delta=args.delta)
    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['graph'],
        replay.dummy['mask']
    )
    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # Training loop
    indices = None
    observations = env.reset()
    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs = gflownet.act(params, key, observations, epsilon)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples)

                train_steps = iteration - args.prefill
                if (train_steps + 1) % (args.log_every * 10) == 0:
                    wandb.log({
                        'replay/delta_scores': wandb.Histogram(replay.transitions['delta_scores']),
                        'replay/scores': wandb.Histogram(replay.transitions['scores']),
                        'replay/num_edges': wandb.Histogram(replay.transitions['num_edges']),
                        'replay/is_exploration': np.mean(replay.transitions['is_exploration']),
                    }, commit=False)
                if (train_steps + 1) % args.log_every == 0:
                    wandb.log({
                        'step': train_steps,
                        'loss': logs['loss'],
                        'replay/size': len(replay),
                        'epsilon': epsilon,

                        'error/mean': jnp.abs(logs['error']).mean(),
                        'error/max': jnp.abs(logs['error']).max(),
                    })
                pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

    # Evaluate the posterior estimate
    posterior, _ = posterior_estimate(
        gflownet,
        params,
        env,
        key,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )
    print('posterior estimate adjacency matrices')
    print(posterior)

    # # Assuming you have a NumPy array with shape (1000, 3, 3)
    # # Replace 'your_array' with the actual array you have
    # your_array = np.random.randint(2, size=(1000, 3, 3))

    # # Reshape the array to (1000, 9) to treat each row as a flattened 3x3 matrix
    # reshaped_array = your_array.reshape((1000, 9))

    # # Use numpy.unique to find unique rows along axis 0
    # unique_rows, unique_counts = np.unique(reshaped_array, axis=0, return_counts=True)

    # # Count the number of unique 3x3 adjacency matrices
    # num_unique_matrices = len(unique_rows)

    # print("Number of unique adjacency matrices:", num_unique_matrices)

    # Compute the metrics
    #ground_truth = nx.to_numpy_array(graph, weight=None)
    wandb.run.summary.update({
        #'metrics/shd/mean': expected_shd(posterior, ground_truth),
        'metrics/edges/mean': expected_edges(posterior),
        #'metrics/thresholds': threshold_metrics(posterior, ground_truth)
    })

    # Save model, data & results
    data.to_csv(os.path.join(wandb.run.dir, 'data.csv'))
    wandb.save('data.csv', policy='now')

    with open(os.path.join(wandb.run.dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    wandb.save('graph.pkl', policy='now')

    io.save(os.path.join(wandb.run.dir, 'model.npz'), params=params)
    wandb.save('model.npz', policy='now')

    replay.save(os.path.join(wandb.run.dir,  'replay_buffer.npz'))
    wandb.save('replay_buffer.npz', policy='now')

    np.save(os.path.join(wandb.run.dir, 'posterior.npy'), posterior)
    wandb.save('posterior.npy', policy='now')

    


    # For small enough graphs, evaluate the full posterior
    # if (args.graph in ['erdos_renyi_lingauss']) or (args.num_variables < 6):
    #     log_features = get_log_features(posterior, data.columns)
    #     full_posterior = get_full_posterior(data, scorer, verbose=True)
    #     full_posterior.save(os.path.join(wandb.run.dir, 'posterior_full.npz'))
    #     wandb.save('posterior_full.npz', policy='now')
        #graphs = full_posterior.graphs
        #nx_graph = nx.DiGraph(graphs[0].edges())
        #nx.draw(nx_graph, with_labels=True)
        #plt.plot()
        #wandb.log({"chart": plt})
        # full_edge_log_features = get_edge_log_features(full_posterior)
        # full_path_log_features = get_path_log_features(full_posterior)
        # full_markov_log_features = get_markov_blanket_log_features(full_posterior)

        # wandb.log({
        #     'posterior/scatter/edge': scatter_from_dicts('full', full_edge_log_features,
        #         'estimate', log_features.edge, transform=np.exp, title='Edge features'),
        #     'posterior/scatter/path': scatter_from_dicts('full', full_path_log_features,
        #         'estimate', log_features.path, transform=np.exp, title='Path features'),
        #     'posterior/scatter/markov_blanket': scatter_from_dicts('full', full_markov_log_features,
        #         'estimate', log_features.markov_blanket, transform=np.exp, title='Markov blanket features')
        # })

    
if __name__ == '__main__':

    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
        help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--log_every', type=int, default=50,
        help='Frequency for logging (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
        help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
        help='Number of samples')
    

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser('sachs_continuous')

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser('sachs_interventional')

    grid_world = subparsers.add_parser('grid_world')
    grid_world.add_argument('--num_variables', type=int, default=9,
        help='Number of variables')
    grid_world.add_argument('--num_edges', type=int, default=6,
        help='Average number of edges')
    grid_world.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    
    coin_flip_bayes = subparsers.add_parser('coin_flip_bayes')
    coin_flip_bayes.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    coin_flip_bayes.add_argument('--num_variables', type=int, default=4,
        help='Number of variables')

    rain_grass_wet = subparsers.add_parser('rain_grass_wet')
    rain_grass_wet.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    rain_grass_wet.add_argument('--num_variables', type=int, default=3,
        help='Number of variables')

    key_door_goal = subparsers.add_parser('key_door_goal')
    key_door_goal.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    key_door_goal.add_argument('--num_variables', type=int, default=3,
        help='Number of variables')

    prototypes = subparsers.add_parser('prototypes')
    prototypes.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    prototypes.add_argument('--num_variables', type=int, default=5,
        help='Number of variables')

    blicket = subparsers.add_parser('blicket')
    blicket.add_argument('--num_samples', type=int, default=1000,
        help='Number of samples')
    blicket.add_argument('--num_variables', type=int, default=4,
        help='Number of variables')
    
    args = parser.parse_args()


    main(args)
