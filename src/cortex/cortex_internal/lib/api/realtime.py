# Copyright 2022 Cortex Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imp
import inspect
import os
from copy import deepcopy
from typing import List, Optional, Union, Any

import dill

from cortex_internal.lib.api.utils import model_downloader
from cortex_internal.lib.api.validations import (
    validate_class_impl,
    validate_python_handler_with_models,
    validate_handler_with_grpc,
    are_models_specified,
)
from cortex_internal.lib.client.python import ModelClient
from cortex_internal.lib.client.tensorflow import TensorFlowClient
from cortex_internal.lib.exceptions import CortexException, UserException, UserRuntimeException
from cortex_internal.lib.model import (
    FileBasedModelsGC,
    TFSAPIServingThreadUpdater,
    ModelsGC,
    ModelTreeUpdater,
    ModelsHolder,
    ModelsTree,
)
from cortex_internal.lib.type import (
    handler_type_from_server_config,
    PythonHandlerType,
    TensorFlowHandlerType,
)

PYTHON_CLASS_VALIDATION = {
    "http": {
        "required": [
            {
                "name": "__init__",
                "required_args": ["self", "config"],
                "optional_args": ["model_client"],
            },
        ],
        "optional": [
            {
                "name": [
                    "handle_post",
                    "handle_get",
                    "handle_put",
                    "handle_patch",
                    "handle_delete",
                ],
                "required_args": ["self"],
                "optional_args": ["payload", "query_params", "headers"],
            },
            {
                "name": "load_model",
                "required_args": ["self", "model_path"],
            },
        ],
    },
    "grpc": {
        "required": [
            {
                "name": "__init__",
                "required_args": ["self", "config", "proto_module_pb2"],
                "optional_args": ["model_client"],
            },
        ],
        "optional": [
            {
                "name": "load_model",
                "required_args": ["self", "model_path"],
            },
        ],
    },
}

TENSORFLOW_CLASS_VALIDATION = {
    "http": {
        "required": [
            {
                "name": "__init__",
                "required_args": ["self", "config", "tensorflow_client"],
                "optional_args": [],
            },
        ],
        "optional": [
            {
                "name": [
                    "handle_post",
                    "handle_get",
                    "handle_put",
                    "handle_patch",
                    "handle_delete",
                ],
                "required_args": ["self"],
                "optional_args": ["payload", "query_params", "headers"],
            },
        ],
    },
    "grpc": {
        "required": [
            {
                "name": "__init__",
                "required_args": ["self", "config", "proto_module_pb2", "tensorflow_client"],
                "optional_args": [],
            },
        ],
    },
}


class RealtimeAPI:
    """
    Class to validate/load the handler class (Handler).
    Also makes the specified models in cortex.yaml available to the handler's implementation.
    """

    def __init__(self, model_server_config: dict, model_dir: str):
        self._model_server_config = model_server_config
        self.model_dir = model_dir

        self.type = handler_type_from_server_config(model_server_config)
        self.path = model_server_config["path"]
        self.config = model_server_config.get("config", {})
        self.protobuf_path = model_server_config.get("protobuf_path")

        self.crons = []
        if not are_models_specified(self._model_server_config):
            return

        self.caching_enabled = self._is_model_caching_enabled()
        self.multiple_processes = self._model_server_config["processes"] > 1

        # model caching can only be enabled when processes is 1
        # model side-reloading is supported for any number of processes

        if self.caching_enabled:
            if self.type == PythonHandlerType:
                mem_cache_size = self._model_server_config["multi_model_reloading"]["cache_size"]
                disk_cache_size = self._model_server_config["multi_model_reloading"][
                    "disk_cache_size"
                ]
            else:
                mem_cache_size = self._model_server_config["models"]["cache_size"]
                disk_cache_size = self._model_server_config["models"]["disk_cache_size"]
            self.models = ModelsHolder(
                self.type,
                self.model_dir,
                mem_cache_size=mem_cache_size,
                disk_cache_size=disk_cache_size,
                on_download_callback=model_downloader,
            )
        elif not self.caching_enabled and self.type != TensorFlowHandlerType:
            self.models = ModelsHolder(self.type, self.model_dir)
        else:
            self.models = None

        if self.multiple_processes:
            self.models_tree = None
        else:
            self.models_tree = ModelsTree()

    @property
    def python_server_side_batching_enabled(self):
        return (
            self._model_server_config.get("server_side_batching") is not None
            and self._model_server_config["type"] == "python"
        )

    def initialize_client(
        self, tf_serving_host: Optional[str] = None, tf_serving_port: Optional[str] = None
    ) -> Union[ModelClient, TensorFlowClient]:
        """
        Initialize client that gives access to models specified in the API spec (cortex.yaml).
        Only applies when models are provided in the API spec.

        Args:
            tf_serving_host: Host of TF serving server. To be only used when the TensorFlow type is used.
            tf_serving_port: Port of TF serving server. To be only used when the TensorFlow type is used.

        Return:
            The client for the respective handler type.
        """

        client = None

        if are_models_specified(self._model_server_config):
            if self.type == PythonHandlerType:
                client = ModelClient(
                    self._model_server_config, self.models, self.model_dir, self.models_tree
                )

            if self.type == TensorFlowHandlerType:
                tf_serving_address = tf_serving_host + ":" + tf_serving_port
                client = TensorFlowClient(
                    tf_serving_address,
                    self._model_server_config,
                    self.models,
                    self.model_dir,
                    self.models_tree,
                )
                if not self.caching_enabled:
                    cron = TFSAPIServingThreadUpdater(interval=5.0, client=client)
                    cron.start()

        return client

    def initialize_impl(
        self,
        project_dir: str,
        client: Union[ModelClient, TensorFlowClient],
        proto_module_pb2: Optional[Any] = None,
        rpc_method_names: Optional[List[str]] = None,
    ):
        """
        Initialize handler class as provided by the user.

        proto_module_pb2 is a module of the compiled proto when grpc is enabled for the "RealtimeAPI" kind. Otherwise, it's None.
        rpc_method_names is a non-empty list when grpc is enabled for the "RealtimeAPI" kind. Otherwise, it's None.

        Can raise UserRuntimeException/UserException/CortexException.
        """

        # build args
        class_impl = self.class_impl(project_dir, rpc_method_names)
        constructor_args = inspect.getfullargspec(class_impl.__init__).args
        config = deepcopy(self.config)
        args = {}
        if "config" in constructor_args:
            args["config"] = config
        if "proto_module_pb2" in constructor_args:
            args["proto_module_pb2"] = proto_module_pb2

        # initialize handler class
        try:
            if self.type == PythonHandlerType:
                if are_models_specified(self._model_server_config):
                    args["model_client"] = client
                    # set load method to enable the use of the client in the constructor
                    # setting/getting from self in load_model won't work because self will be set to None
                    client.set_load_method(
                        lambda model_path: class_impl.load_model(None, model_path)
                    )
                    initialized_impl = class_impl(**args)
                    client.set_load_method(initialized_impl.load_model)
                else:
                    initialized_impl = class_impl(**args)
            if self.type == TensorFlowHandlerType:
                args["tensorflow_client"] = client
                initialized_impl = class_impl(**args)
        except Exception as e:
            raise UserRuntimeException(self.path, "__init__", str(e)) from e

        # initialize the crons if models have been specified and if the API kind is RealtimeAPI
        if are_models_specified(self._model_server_config):
            if not self.multiple_processes and self.caching_enabled:
                self.crons += [
                    ModelTreeUpdater(
                        interval=10,
                        model_server_config=self._model_server_config,
                        tree=self.models_tree,
                        ondisk_models_dir=self.model_dir,
                    ),
                    ModelsGC(
                        interval=10,
                        model_server_config=self._model_server_config,
                        models=self.models,
                        tree=self.models_tree,
                    ),
                ]

            if not self.caching_enabled and self.type == PythonHandlerType:
                self.crons += [
                    FileBasedModelsGC(interval=10, models=self.models, download_dir=self.model_dir)
                ]

        for cron in self.crons:
            cron.start()

        return initialized_impl

    def class_impl(self, project_dir: str, rpc_method_names: Optional[List[str]] = None):
        """Can only raise UserException/CortexException exceptions"""
        target_class_name = "Handler"
        if self.type == TensorFlowHandlerType:
            validations = TENSORFLOW_CLASS_VALIDATION
        elif self.type == PythonHandlerType:
            validations = PYTHON_CLASS_VALIDATION
        else:
            raise CortexException(f"invalid handler type: {self.type}")

        try:
            handler_class = self._get_class_impl(
                "cortex_handler", os.path.join(project_dir, self.path), target_class_name
            )
        except Exception as e:
            e.wrap("error in " + self.path)
            raise

        try:
            validate_class_impl(handler_class, validations)
            validate_handler_with_grpc(handler_class, self._model_server_config, rpc_method_names)
            if self.type == PythonHandlerType:
                validate_python_handler_with_models(handler_class, self._model_server_config)
        except Exception as e:
            e.wrap("error in " + self.path)
            raise
        return handler_class

    def _get_class_impl(self, module_name, impl_path, target_class_name):
        """Can only raise UserException exception"""
        if impl_path.endswith(".pickle"):
            try:
                with open(impl_path, "rb") as pickle_file:
                    return dill.load(pickle_file)
            except Exception as e:
                raise UserException("unable to load pickle", str(e)) from e

        try:
            impl = imp.load_source(module_name, impl_path)
        except Exception as e:
            raise UserException(str(e)) from e

        classes = inspect.getmembers(impl, inspect.isclass)
        handler_class = None
        for class_df in classes:
            if class_df[0] == target_class_name:
                if handler_class is not None:
                    raise UserException(
                        f"multiple definitions for {target_class_name} class found; please check your imports and class definitions and ensure that there is only one handler class definition"
                    )
                handler_class = class_df[1]
        if handler_class is None:
            raise UserException(f"{target_class_name} class is not defined")

        return handler_class

    def _is_model_caching_enabled(self) -> bool:
        """
        Checks if model caching is enabled.
        """
        models = None
        if self.type == TensorFlowHandlerType and "models" in self._model_server_config:
            models = self._model_server_config["models"]
        if self.type == PythonHandlerType and "multi_model_reloading" in self._model_server_config:
            models = self._model_server_config["multi_model_reloading"]

        return models and models["cache_size"] and models["disk_cache_size"]

    def __del__(self) -> None:
        for cron in self.crons:
            cron.stop()
        for cron in self.crons:
            cron.join()
