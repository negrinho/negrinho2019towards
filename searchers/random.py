from searchers.common import random_specify, Searcher


class RandomSearcher(Searcher):

    def __init__(self, search_space_fn):
        Searcher.__init__(self, search_space_fn)

    def sample(self):
        inputs, outputs = self.search_space_fn()
        while True:
            try:
                vs = random_specify(outputs.values())
                return inputs, outputs, vs, {}
            except ValueError:
                inputs, outputs = self.search_space_fn()

    def update(self, val, searcher_eval_token):
        pass

    def save_state(self, folder):
        pass

    def load_state(self, folder):
        pass
