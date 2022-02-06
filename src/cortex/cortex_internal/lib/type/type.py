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

import collections


class HandlerType(collections.namedtuple("HandlerType", "type")):
    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return str(self.type)


PythonHandlerType = HandlerType("python")
TensorFlowHandlerType = HandlerType("tensorflow")


def handler_type_from_string(handler_type: str) -> HandlerType:
    """
    Get handler type from string.

    Args:
        handler_type: "python" or "tensorflow"

    Raises:
        ValueError if handler_type does not hold the right value.
    """
    handler_types = [
        PythonHandlerType,
        TensorFlowHandlerType,
    ]
    for candidate in handler_types:
        if str(candidate) == handler_type:
            return candidate
    raise ValueError("handler_type can only be 'python' or 'tensorflow'")


def handler_type_from_server_config(model_server_config: dict) -> HandlerType:
    """
    Get handler type from model server config.
    """
    return handler_type_from_string(model_server_config["type"])
