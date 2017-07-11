#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeManagerWorkerPyro",)

import time

from pyomo.pysp.util.misc import _EnumValueWithData
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.scenariotree.server_pyro_utils import \
    WorkerInitType
from pyomo.pysp.scenariotree.manager \
    import (_ScenarioTreeManagerWorker,
            ScenarioTreeManager,
            InvocationType)

import six
from six import iteritems, string_types

try:
    import dill
    dill_available = True
except ImportError:                               #pragma:nocover
    dill_available = False

#
# A full implementation of the ScenarioTreeManager interface
# designed to be used by Pyro-based ScenarioTreeManagerClient
# implementations.
#

class ScenarioTreeManagerWorkerPyro(_ScenarioTreeManagerWorker,
                                    ScenarioTreeManager,
                                    PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerWorkerPyro class")

    #
    # scenario instance construction
    #
    safe_declare_common_option(_declared_options,
                               "objective_sense_stage_based")
    safe_declare_common_option(_declared_options,
                               "output_instance_construction_time")
    safe_declare_common_option(_declared_options,
                               "compile_scenario_instances")

    #
    # various
    #
    safe_declare_common_option(_declared_options,
                               "verbose")
    safe_declare_common_option(_declared_options,
                               "profile_memory")

    def __init__(self, *args, **kwds):
        super(ScenarioTreeManagerWorkerPyro, self).__init__(*args, **kwds)

        self._modules_imported = None
        # The name of the scenario tree server owning this worker
        self._server_name = None
        # The name of the worker on this server
        self._worker_name = None
        # a reference to the mpi4py namespace
        self.MPI = None
        # a dict of intracommunicators for each tree node
        self.mpi_comm_tree = {}

    def _collect_scenario_tree_data_for_client(self, tree_object_names):

        data = {}
        node_data = data['nodes'] = {}
        for node_name in tree_object_names['nodes']:
            tree_node = self._scenario_tree.get_node(node_name)
            this_node_data = node_data[node_name] = {}
            this_node_data['_variable_ids'] = tree_node._variable_ids
            this_node_data['_standard_variable_ids'] = \
                tree_node._standard_variable_ids
            this_node_data['_variable_indices'] = tree_node._variable_indices
            this_node_data['_integer'] = tuple(tree_node._integer)
            this_node_data['_binary'] = tuple(tree_node._binary)
            this_node_data['_semicontinuous'] = \
                tuple(tree_node._semicontinuous)
            # master will need to reconstruct
            # _derived_variable_ids
            # _name_index_to_id

        scenario_data = data['scenarios'] = {}
        for scenario_name in tree_object_names['scenarios']:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            this_scenario_data = scenario_data[scenario_name] = {}
            this_scenario_data['_objective_name'] = scenario._objective_name
            this_scenario_data['_objective_sense'] = \
                scenario._objective_sense

        return data

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _init(self,
              server_name,
              uncompressed_scenario_tree,
              worker_name,
              worker_init,
              modules_imported,
              MPI,
              root_comm):
        # check to make sure no base class has implemented _init
        try:
            super(ScenarioTreeManagerWorkerPyro, self)._init()
        except NotImplementedError:
            pass
        else:
            assert False, "developer error"

        self._modules_imported = modules_imported
        # The name of the scenario tree server owning this worker
        self._server_name = server_name
        # So we have access to real scenario and bundle probabilities
        self._uncompressed_scenario_tree = uncompressed_scenario_tree
        self._worker_name = worker_name

        scenarios_to_construct = []
        if worker_init.type_ == WorkerInitType.Scenarios:
            assert type(worker_init.names) in (list, tuple)
            assert len(worker_init.names) > 0
            assert worker_init.data is None

            if self._options.verbose:
                print("Constructing worker with name %s for scenarios: %s"
                      % (worker_name, str(worker_init.names)))

            scenarios_to_construct.extend(worker_init.names)

        elif worker_init.type_ == WorkerInitType.Bundles:
            assert type(worker_init.names) in (list, tuple)
            assert type(worker_init.data) is dict
            assert len(worker_init.names) > 0

            if self._options.verbose:
                print("Constructing worker with name %s for bundle list:"
                      % (worker_name))
                for bundle_name in worker_init.names:
                    assert type(worker_init.data[bundle_name]) in (list, tuple)
                    print("  - %s: %s" % (bundle_name, worker_init.data[bundle_name]))

            for bundle_name in worker_init.names:
                assert type(worker_init.data[bundle_name]) in (list, tuple)
                scenarios_to_construct.extend(worker_init.data[bundle_name])

        # compress the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree = \
            self.uncompressed_scenario_tree.make_compressed(
                scenarios_to_construct,
                normalize=False)
        self._instances = \
            self.uncompressed_scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                output_instance_construction_time=\
                   self._options.output_instance_construction_time,
                profile_memory=self._options.profile_memory,
                compile_scenario_instances=self._options.compile_scenario_instances)

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self._options.objective_sense_stage_based,
            create_variable_ids=True)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        #
        # Create bundle if needed
        #
        if worker_init.type_ == WorkerInitType.Bundles:
            for bundle_name in worker_init.names:
                assert not self._scenario_tree.contains_bundle(bundle_name)
                self.add_bundle(bundle_name, worker_init.data[bundle_name])
                assert self._scenario_tree.contains_bundle(bundle_name)

        # now generate the process communicators
        if MPI is None:
            assert root_comm is None
        else:
            assert root_comm is not None
            root_node = self._scenario_tree.findRootNode()
            self.MPI = MPI
            self.mpi_comm_tree[root_node.name] = root_comm
            # loop over all nodes except the root and leaf
            # nodes and create a communicator between all
            # processes that reference a node
            for stage in self.uncompressed_scenario_tree.stages[1:-1]:
                for node in stage.nodes:
                    if self._scenario_tree.contains_node(node.name):
                        self.mpi_comm_tree[node.name] = \
                            self.mpi_comm_tree[node.parent.name].\
                            Split(0)
                    elif node.parent.name in self.mpi_comm_tree:
                        self.mpi_comm_tree[node.parent.name].\
                            Split(self.MPI.UNDEFINED)

    # Override the implementation on _ScenarioTreeManagerWorker
    def _close_impl(self):
        super(ScenarioTreeManagerWorkerPyro, self)._close_impl()
        ignored_options = dict((_c._name, _c.value(False))
                               for _c in self._options.unused_user_values())
        for comm in self.mpi_comm_tree.values():
            comm.Free()
        if len(ignored_options):
            print("")
            print("*** WARNING: The following options were explicitly "
                  "set but never accessed by worker %s: "
                  % (self._worker_name))
            for name in ignored_options:
                print(" - %s: %s" % (name, ignored_options[name]))
            print("*** If you believe this is a bug, please report it "
                  "to the PySP developers.")
            print("")

    def _invoke_function_impl(self,
                              function,
                              module_name=None,
                              invocation_type=InvocationType.Single,
                              function_args=(),
                              function_kwds=None):

        start_time = time.time()

        if self._options.verbose:
            if module_name is not None:
                print("Received request to invoke function=%s "
                      "in module=%s" % (str(function), str(module_name)))
            else:
                print("Received request to invoke anonymous "
                      "function serialized using the dill module")

        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just transmit it
        # as a (key, data) tuple in that case
        if type(invocation_type) is tuple:
            _invocation_type_key, _invocation_type_data = invocation_type
            assert isinstance(_invocation_type_key, string_types)
            invocation_type = getattr(InvocationType,
                                      _invocation_type_key)
            if _invocation_type_data is not None:
                assert isinstance(invocation_type, _EnumValueWithData)
                invocation_type = invocation_type(_invocation_type_data)

        # here we assume that if the module_name is None,
        # then a function was serialized by the fill module
        # before being transmitted
        if module_name is None:
            assert dill_available
            function = dill.loads(function)

        result = self._invoke_function_by_worker(
            function,
            module_name=module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_impl(self,
                            method_name,
                            method_args=(),
                            method_kwds=None):

        start_time = time.time()

        if self._options.verbose:
            print("Received request to invoke method="+method_name)

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    # implemented by _ScenarioTreeManagerWorker
    #def _add_bundle_impl(...)

    # implemented by _ScenarioTreeManagerWorker
    #def _remove_bundle_impl(...)

    #
    # Override the invoke_function and invoke_method interface methods
    # on ScenarioTreeManager
    # ** NOTE **: These version are meant to be invoked locally.
    #             The client-side will always invoke the *_impl
    #             methods, which do not accept the async or
    #             oneway keywords. When invoked here, the
    #             async and oneway keywords behave like they
    #             do for the Serial solver manager (they are
    #             a dummy interface)
    #

    def invoke_function(self,
                        function,
                        module_name=None,
                        invocation_type=InvocationType.Single,
                        function_args=(),
                        function_kwds=None,
                        async=False,
                        oneway=False):
        """This function is an override of that on the
        ScenarioTreeManager interface. It should not be invoked by a
        client, but only locally (e.g., inside a local function
        invocation transmitted by the client).
        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        if not isinstance(function, six.string_types):
            if module_name is not None:
                raise ValueError(
                    "The module_name keyword must be None "
                    "when the function argument is not a string.")
        else:
            if module_name is None:
                raise ValueError(
                    "A module name is required when "
                    "a function name is given")

        self._invoke_function_impl(function,
                                   module_name=module_name,
                                   invocation_type=invocation_type,
                                   function_args=function_args,
                                   function_kwds=function_kwds)

        if not oneway:
            if invocation_type == InvocationType.Single:
                result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(None, result=result)

        return result

    def invoke_method(self,
                      method_name,
                      method_args=(),
                      method_kwds=None,
                      async=False,
                      oneway=False):
        """This function is an override of that on the
        ScenarioTreeManager interface. It should not be invoked by a
        client, but only locally (e.g., inside a local function
        invocation transmitted by the client).

        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        if not oneway:
            result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(None, result=result)

        return result

    #
    # Helper methods that can be invoked by the client
    #

    def assign_data(self, name, data):
        if self._options.verbose:
            print("Received request to assign data to attribute name %s on "
                  "scenario tree worker %s" % (name, self._worker_name))
        setattr(self, name, data)
