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


def extract_from_handler(server_config: dict) -> dict:
    handler_type = server_config["type"].lower()

    env_vars = {
        "CORTEX_LOG_LEVEL": server_config["log_level"].upper(),
        "CORTEX_SERVING_PORT": server_config["serve_port"],
        "CORTEX_PROCESSES": server_config["processes"],
        "CORTEX_THREADS_PER_PROCESS": server_config["threads_per_process"],
        "CORTEX_DEPENDENCIES_PIP": server_config["dependencies"]["pip"],
        "CORTEX_DEPENDENCIES_CONDA": server_config["dependencies"]["conda"],
        "CORTEX_DEPENDENCIES_SHELL": server_config["dependencies"]["shell"],
        "CORTEX_MAX_CONCURRENCY": server_config["max_concurrency"],
    }

    env_vars["CORTEX_PYTHON_PATH"] = os.path.normpath(
        os.path.join("/src", "project", server_config["python_path"])
    )

    if server_config.get("protobuf_path") is not None:
        env_vars["CORTEX_SERVING_PROTOCOL"] = "grpc"
        env_vars["CORTEX_PROTOBUF_FILE"] = os.path.join(
            "/src", "project", server_config["protobuf_path"]
        )
    else:
        env_vars["CORTEX_SERVING_PROTOCOL"] = "http"

    if handler_type == "tensorflow":
        env_vars["CORTEX_TF_BASE_SERVING_PORT"] = "9000"

    for key, val in server_config["env"].items():
        env_vars[key] = val

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
        "CORTEX_PROJECT_DIR": "/src/project",
        "CORTEX_MODEL_DIR": "/mnt/model",
        "CORTEX_LOG_CONFIG_FILE": "/src/cortex/log_config.yaml",
        "CORTEX_PYTHON_PATH": "/src/project",
    }
    env_vars.update(extract_from_handler(server_config))

    set_env_vars_for_s6(env_vars)
    print_env_var_exports(env_vars)


if __name__ == "__main__":
    main(sys.argv[1])
