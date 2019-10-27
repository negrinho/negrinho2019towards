"""
Implementation of Regularized Evolution searcher from Zoph et al '18
"""
from __future__ import print_function

from builtins import range
from collections import deque
import sys
random = sys.modules['random']

from deep_architect.utils import join_paths, write_jsonfile, read_jsonfile, file_exists
from deep_architect.core import unassigned_independent_hyperparameter_iterator
from searchers.common import Searcher, random_specify_hyperparameter


def mutatable(h):
    return len(h.vs) > 1


def mutate(output_lst, user_vs, all_vs, mutatable_fn, search_space_fn):
    mutate_candidates = []
    new_vs = list(user_vs)
    for i, h in enumerate(
            unassigned_independent_hyperparameter_iterator(output_lst)):
        if mutatable_fn(h):
            mutate_candidates.append(h)
        h.assign_value(all_vs[i])

    # mutate a random hyperparameter
    assert len(mutate_candidates) == len(user_vs)
    m_ind = random.randint(0, len(mutate_candidates) - 1)
    m_h = mutate_candidates[m_ind]
    v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]

    # ensure that same value is not chosen again
    while v == user_vs[m_ind]:
        v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]
    new_vs[m_ind] = v
    if 'sub' in m_h.get_name():
        new_vs = new_vs[:m_ind + 1]

    inputs, outputs = search_space_fn()
    output_lst = list(outputs.values())
    all_vs = specify_evolution(output_lst, mutatable_fn, new_vs)
    return inputs, outputs, new_vs, all_vs


def random_specify_evolution(output_lst, mutatable_fn):
    user_vs = []
    all_vs = []
    for h in unassigned_independent_hyperparameter_iterator(output_lst):
        v = random_specify_hyperparameter(h)
        if mutatable_fn(h):
            user_vs.append(v)
        all_vs.append(v)
    return user_vs, all_vs


def specify_evolution(output_lst, mutatable_fn, user_vs):
    vs_idx = 0
    vs = []
    for i, h in enumerate(
            unassigned_independent_hyperparameter_iterator(output_lst)):
        if mutatable_fn(h):
            if vs_idx >= len(user_vs):
                user_vs.append(h.vs[random.randint(0, len(h.vs) - 1)])
            h.assign_value(user_vs[vs_idx])
            vs.append(user_vs[vs_idx])
            vs_idx += 1
        else:
            v = random_specify_hyperparameter(h)
            vs.append(v)
    return vs


class EvolutionSearcher(Searcher):

    def __init__(self, search_space_fn, mutatable_fn, P, S, regularized=False):
        Searcher.__init__(self, search_space_fn)
        # Population size
        self.P = P
        # Sample size
        self.S = S

        self.population = deque(maxlen=P)
        self.regularized = regularized
        self.initializing = True
        self.mutatable = mutatable_fn

    def sample(self):
        if self.initializing:
            inputs, outputs = self.search_space_fn()
            user_vs, all_vs = random_specify_evolution(list(outputs.values()),
                                                       self.mutatable)
            if len(self.population) >= self.P - 1:
                self.initializing = False
            return inputs, outputs, all_vs, {
                'user_vs': user_vs,
                'all_vs': all_vs
            }
        else:
            sample_inds = sorted(
                random.sample(list(range(len(self.population))),
                              min(self.S, len(self.population))))

            # mutate strongest model
            inputs, outputs = self.search_space_fn()
            user_vs, all_vs, _ = self.population[self.get_strongest_model_index(
                sample_inds)]
            inputs, outputs, new_user_vs, new_all_vs = mutate(
                list(outputs.values()), user_vs, all_vs, self.mutatable,
                self.search_space_fn)

            return inputs, outputs, new_all_vs, {
                'user_vs': new_user_vs,
                'all_vs': new_all_vs
            }

    def update(self, val, cfg_d):
        if not self.initializing:
            weak_ind = self.get_weakest_model_index()
            del self.population[weak_ind]
        self.population.append((cfg_d['user_vs'], cfg_d['all_vs'], val))

    def get_weakest_model_index(self):
        if self.regularized:
            return 0
        else:
            min_acc = 1.
            min_acc_ind = -1
            for i in range(len(self.population)):
                _, _, acc = self.population[i]
                if acc < min_acc:
                    min_acc = acc
                    min_acc_ind = i
            return min_acc_ind

    def get_strongest_model_index(self, sample_inds):
        max_acc = 0.
        max_acc_ind = -1
        for i in range(len(sample_inds)):
            _, _, acc = self.population[sample_inds[i]]
            if acc > max_acc:
                max_acc = acc
                max_acc_ind = i
        return sample_inds[max_acc_ind]

    def get_searcher_state_token(self):
        return {
            "P": self.P,
            "S": self.S,
            "population": list(self.population),
            "regularized": self.regularized,
            "initializing": self.initializing,
        }

    def save_state(self, folder_name):
        state = self.get_searcher_state_token()
        write_jsonfile(state,
                       join_paths([folder_name, 'evolution_searcher.json']))

    def load_state(self, folder_name):
        filepath = join_paths([folder_name, 'evolution_searcher.json'])
        if not file_exists(filepath):
            raise RuntimeError("Load file does not exist")

        state = read_jsonfile(filepath)
        self.P = state["P"]
        self.S = state["S"]
        self.regularized = state['regularized']
        self.population = deque(state['population'])
        self.initializing = state['initializing']