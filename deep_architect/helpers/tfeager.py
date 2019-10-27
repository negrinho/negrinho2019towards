import numpy as np
import tensorflow as tf

import deep_architect.core as co
from deep_architect.hyperparameters import D


class TFEModule(co.Module):
    """Class for taking TFEager code and wrapping it in a DeepArchitect module.

    This class subclasses :class:`deep_architect.core.Module` as therefore inherits all
    the information associated to it (e.g., inputs, outputs, and hyperparameters).
    It also enables to do the compile and forward operations for these types of
    modules once a module is fully specified, i.e., once all the hyperparameters
    have been chosen.

    The compile operation in this case instantiates any TFEager variables necessary
    for the computation associated to this module.
    The forward operation takes the variables that were created in the compile
    operation and constructs the actual computational graph fragment associated
    to this module.

    See :class:`deep_architect.helpers.tensorflow_support.TensorflowModule` for a similar class for
    Tensorflow. One of the main differences is that Tensorflow deals with
    static computational graphs, so the forward functionality is usually only
    called once per creation for the graph creation. TFEager requires calling
    forward for each tensor of data that is fed through the network.

    .. note::
        This module is abstract, meaning that it does not actually implement
        any particular TFEager computation. It simply wraps TFEager
        functionality in a DeepArchitect module. This functionality makes extensive use
        of closures.

        The keys of the dictionaries that are passed to the compile function
        match the names of the inputs and hyperparameters, respectively.
        The keys of the dictionary that are passed to the forward function match
        the names of the inputs. The keys of dictionary returned by the forward
        function match the names of the outputs.

    Args:
        name (str): Name of the module
        name_to_hyperp (dict[str,deep_architect.core.Hyperparameter]): Dictionary of
            hyperparameters that the model depends on. The keys are the local
            names of the hyperparameters.
        compile_fn ((dict[str,object], dict[str,object]) -> ((dict[str,object]) -> (dict[str,object], list[torch.nn.Modules]))):
            The first function takes two dictionaries with
            keys corresponding to `input_names` and `output_names` and returns
            a function that takes a dictionary with keys corresponding to
            `input_names` and returns a dictionary with keys corresponding
            to `output_names` and a list of Pytorch modules involved in the
            computation of the DeepArchitect module.
        input_names (list[str]): List of names for the inputs.
        output_names (list[str]): List of names for the outputs.
        scope (deep_architect.core.Scope, optional): Scope where the module will be
            registered.
    """

    def __init__(self,
                 name,
                 name_to_hyperp,
                 compile_fn,
                 input_names,
                 output_names,
                 scope=None):
        co.Module.__init__(self, scope, name)
        for h in name_to_hyperp:
            if not isinstance(name_to_hyperp[h], co.Hyperparameter):
                vs = name_to_hyperp[h] if isinstance(
                    name_to_hyperp[h], list) else [name_to_hyperp[h]]
                name_to_hyperp[h] = D(vs)
        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn
        self.isTraining = True

    def _compile(self):
        input_name_to_val = self._get_input_values()
        hyperp_name_to_val = self._get_hyperp_values()
        self._fn = self._compile_fn(input_name_to_val, hyperp_name_to_val)

    def _forward(self):
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val,
                                      isTraining=self.isTraining)
        self._set_output_values(output_name_to_val)

    def _update(self):
        pass


def setTraining(output_lst, isTraining):

    def fn(mx):
        if hasattr(mx, 'isTraining'):
            mx.isTraining = isTraining

    co.traverse_backward(output_lst, fn)


def siso_tfeager_module(name, compile_fn, name_to_hyperp, scope=None):
    return TFEModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'],
                     scope).get_io()


def siso_tfeager_module_from_tensorflow_op_fn(layer_fn,
                                              name_to_hyperp,
                                              scope=None,
                                              name=None):

    def compile_fn(di, dh):
        m = layer_fn(**dh)

        def forward_fn(di, isTraining=False):
            return {"Out": m(di["In"])}

        return forward_fn

    if name is None:
        name = layer_fn.__name__

    return siso_tfeager_module(name, compile_fn, name_to_hyperp, scope)


def get_num_trainable_parameters():
    return np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])