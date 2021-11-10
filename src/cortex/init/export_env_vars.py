#!/usr/bin/env bash

# Copyright 2021 Cortex Labs, Inc.
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

import yaml
import os
import sys
from pathlib import Path

NEURON_CORES_PER_INF = 4


def extract_from_handler(server_config: dict) -> dict:
    handler_type = server_config["type"].lower()

    env_vars = {
        "CORTEX_LOG_LEVEL": server_config["log_level"].upper(),
        "CORTEX_PROCESSES_PER_REPLICA": server_config["processes_per_replica"],
        "CORTEX_THREADS_PER_PROCESS": server_config["threads_per_process"],
        "CORTEX_DEPENDENCIES_PIP": server_config["dependencies"]["pip"],
        "CORTEX_DEPENDENCIES_CONDA": server_config["dependencies"]["conda"],
        "CORTEX_DEPENDENCIES_SHELL": server_config["dependencies"]["shell"],
        "CORTEX_MAX_REPLICA_CONCURRENCY": int(server_config["processes_per_replica"])
        * int(server_config["threads_per_process"]),
    }

    env_vars["CORTEX_PYTHON_PATH"] = os.path.normpath(
        os.path.join("/mnt", "project", server_config["python_path"])
    )

    if server_config.get("protobuf_path") is not None:
        env_vars["CORTEX_SERVING_PROTOCOL"] = "grpc"
        env_vars["CORTEX_PROTOBUF_FILE"] = os.path.join(
            "/mnt", "project", server_config["protobuf_path"]
        )
    else:
        env_vars["CORTEX_SERVING_PROTOCOL"] = "http"

    if handler_type == "tensorflow":
        env_vars["CORTEX_TF_BASE_SERVING_PORT"] = "9000"
        env_vars["CORTEX_TF_SERVING_HOST"] = "localhost"

    return env_vars


def set_env_vars_for_s6(env_vars: dict):
    s6_env_base = "/var/run/s6/container_environment"

    Path(s6_env_base).mkdir(parents=True, exist_ok=True)

    for k, v in env_vars.items():
        if v is not None:
            Path(f"{s6_env_base}/{k}").write_text(str(v))


def print_env_var_exports(env_vars: dict):
    for k, v in env_vars.items():
        if v is not None:
            print(f"export {k}='{v}'")


def main(model_server_config_path: str):
    with open(model_server_config_path, "r") as f:
        server_config = yaml.safe_load(f)

    env_vars = {
        "CORTEX_SERVING_PORT": 8888,
        "CORTEX_CACHE_DIR": "/mnt/cache",
        "CORTEX_PROJECT_DIR": "/mnt/project",
        "CORTEX_MODEL_DIR": "/mnt/model",
        "CORTEX_LOG_CONFIG_FILE": "/src/cortex/log_config.yaml",
        "CORTEX_PYTHON_PATH": "/mnt/project",
    }
    env_vars.update(extract_from_handler(server_config))

    set_env_vars_for_s6(env_vars)
    print_env_var_exports(env_vars)


if __name__ == "__main__":
    main(sys.argv[1])
