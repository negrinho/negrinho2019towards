import logging
import argparse

from searchers import random as rs, mcts, regularized_evolution_searcher, smbo_random
from search_spaces import genetic_space, nasbench, nasnet_space, main_hierarchical
from evaluators import tpu_estimator_classification
from surrogates import hashing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)


def run_search(searcher, evaluator, num_samples):
    for evaluation_id in range(num_samples):
        inputs, outputs, vs, sst = searcher.sample()
        results = evaluator.eval(inputs, outputs)
        results = {'validation_accuracy': .2}
        searcher.update(results['validation_accuracy'], sst)
        logger.info('Results evaluation %d:\n\tConfig:%s\n\tResults:%s',
                    evaluation_id, str(vs), str(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-space',
                        choices=['genetic', 'nasnet', 'nasbench', 'flat'],
                        default='genetic')
    parser.add_argument('--searcher',
                        choices=['random', 'mcts', 'smbo', 'evolution'],
                        default='random')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--tpu-name', default='')
    parser.add_argument('--use-tpu', action='store_true')
    parser.add_argument('--evaluation-dir', default='./scratch')
    parser.add_argument('--num-samples', type=int, default=128)

    args = parser.parse_args()

    if args.use_tpu and (args.tpu_name == '' or
                         not args.evaluation_dir.startswith('gs://') or
                         not args.data_dir.startswith('gs://')):
        raise ValueError('If using TPU, TPU arguments need to be provided')

    ssf_fns = {
        'genetic': genetic_space.SSF_Genetic,
        'nasnet': nasnet_space.SSF_NasnetA,
        'nasbench': nasbench.SSF_Nasbench,
        'flat': main_hierarchical.SSF_Flat,
    }

    ssf = ssf_fns[args.search_space]()

    searcher_fns = {
        'random':
        lambda: rs.RandomSearcher(ssf.get_search_space),
        'mcts':
        lambda: mcts.MCTSSearcher(ssf.get_search_space, .33),
        'smbo':
        lambda: smbo_random.SMBOSearcher(
            ssf.get_search_space, hashing.HashingSurrogate(2**16, 1), 512, .1),
        'evolution':
        lambda: regularized_evolution_searcher.EvolutionSearcher(
            ssf.get_search_space,
            regularized_evolution_searcher.mutatable,
            100,
            25,
            regularized=True),
    }
    searcher = searcher_fns[args.searcher]()
    evaluator = tpu_estimator_classification.AdvanceClassifierEvaluator(
        args.data_dir,
        args.tpu_name,
        25,
        base_dir=args.evaluation_dir,
        use_tpu=args.use_tpu)

    run_search(searcher, evaluator, args.num_samples)


if __name__ == '__main__':
    main()
