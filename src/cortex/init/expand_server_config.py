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
import sys


def main(model_server_config_path: str):
    with open(model_server_config_path, "r") as f:
        server_config = yaml.safe_load(f)
    # print(yaml.safe_dump(server_config, indent=2, sort_keys=False))

    models_field_name: str = None
    if "multi_model_reloading" in server_config:
        models_field_name = "multi_model_reloading"
    if "models" in server_config:
        models_field_name = "models"
    if models_field_name:
        if "signature_key" not in server_config[models_field_name]:
            server_config[models_field_name]["signature_key"] = None
        if "paths" in server_config[models_field_name]:
            for idx, model in enumerate(server_config[models_field_name]["paths"]):
                if "signature_key" not in model:
                    server_config[models_field_name][idx]["signature_key"] = None
        if "paths" not in server_config[models_field_name]:
            server_config[models_field_name]["paths"] = None
        if "dir" not in server_config[models_field_name]:
            server_config[models_field_name]["dir"] = None
        if "path" not in server_config[models_field_name]:
            server_config[models_field_name]["path"] = None
        if "cache_size" not in server_config[models_field_name]:
            server_config[models_field_name]["cache_size"] = None
        if "disk_cache_size" not in server_config[models_field_name]:
            server_config[models_field_name]["disk_cache_size"] = None

    print(yaml.safe_dump(server_config, indent=2, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1])
