import pandas as pd
import urllib.request
import gzip
import os 
from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian, sample_grid_world_2x2, sample_grid_world_3x3, sample_grid_world_4x4, sample_coin_flip, sample_rain_grass_wet, sample_key_door_goal
from dag_gflownet.utils.sampling import sample_from_linear_gaussian, sample_from_discrete


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename

def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'

    # elif name == 'grid_world':
    #     graph = sample_grid_world_2x2(rng=rng)
    #     filename = 'traj_matrix_2x2.csv'
    #     data = pd.read_csv(filename, delimiter=',', dtype='category')
    #     score = 'bde'

    elif name == 'grid_world':
        graph = sample_grid_world_4x4(rng=rng)
        filename = 'traj_matrix_4x4.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        score = 'bde'

    elif name == 'coin_flip_bayes':
        graph = sample_coin_flip(num_variables=args.num_variables, rng=rng)
        data = sample_from_discrete(model=graph, num_samples=args.num_samples, rng=rng)
        score = 'bde'

    elif name == 'rain_grass_wet':
        graph = sample_rain_grass_wet(num_variables=args.num_variables, rng=rng)
        data = sample_from_discrete(model=graph, num_samples=args.num_samples, rng=rng)
        score = 'bde'

    elif name == 'key_door_goal':
        filename = 'merged-intvs-doina.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        graph = sample_key_door_goal(num_variables=args.num_variables, rng=rng)
        score = 'bde'
        
    elif name == 'prototypes':
        current_working_directory = os.getcwd()
        print(f"Current Working Directory: {current_working_directory}")
        filename = 'datasets/key_to_door/merged_kd3_14k_5p.csv' #'merged_kd3.csv' #'gfn-kd3-cos.csv'#'gfn-data-kd4-sf.csv'#'conspec-key4-3k.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        graph = sample_key_door_goal(num_variables=args.num_variables, rng=rng)
        score = 'bde'

    elif name == 'blicket':
        filename = 'datasets/blicket_detector/Adisj_new.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        graph = sample_key_door_goal(num_variables=args.num_variables, rng=rng)
        score = 'bde'
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score

