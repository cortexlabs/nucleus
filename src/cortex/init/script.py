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

import yaml
import os
import sys
import time

from cortex_internal.lib.log import configure_logger
from cortex_internal.lib.telemetry import get_default_tags, init_sentry

init_sentry(tags=get_default_tags())
logger = configure_logger("cortex", os.environ["CORTEX_LOG_CONFIG_FILE"])

from cortex_internal.lib.type import (
    handler_type_from_server_config,
    PythonHandlerType,
    TensorFlowHandlerType,
)
from cortex_internal.lib.model import (
    FileBasedModelsTreeUpdater,  # only when num workers > 1
    TFSModelLoader,
)


def are_models_specified(model_server_config: dict) -> bool:
    """
    Checks if models have been specified in the server config (cortex.yaml).

    Args:
        server_config: Model server config.
    """

    return model_server_config["multi_model_reloading"]


def is_model_caching_enabled(model_server_config: dir) -> bool:
    handler_type = handler_type_from_server_config(model_server_config)

    if handler_type == PythonHandlerType and model_server_config["multi_model_reloading"]:
        models = model_server_config["multi_model_reloading"]
    elif handler_type == TensorFlowHandlerType and model_server_config["models"]:
        models = model_server_config["models"]
    else:
        return False

    return models and models["cache_size"] and models["disk_cache_size"]


def main():
    # get API spec
    model_server_config_path = os.environ["CORTEX_MODEL_SERVER_CONFIG"]

    with open(model_server_config_path) as yaml_file:
        model_server_config = yaml.safe_load(yaml_file)

    handler_type = handler_type_from_server_config(model_server_config)
    caching_enabled = is_model_caching_enabled(model_server_config)
    model_dir = os.getenv("CORTEX_MODEL_DIR")

    # start live-reloading when model caching not enabled > 1
    cron = None
    if not caching_enabled:
        # create cron dirs if they don't exist
        os.makedirs("/run/cron", exist_ok=True)
        os.makedirs("/tmp/cron", exist_ok=True)

        # prepare crons
        if handler_type == PythonHandlerType and are_models_specified(model_server_config):
            cron = FileBasedModelsTreeUpdater(
                interval=10,
                model_server_config=model_server_config,
                download_dir=model_dir,
            )
            cron.start()
        elif handler_type == TensorFlowHandlerType:
            tf_serving_port = os.getenv("CORTEX_TF_BASE_SERVING_PORT", "9000")
            tf_serving_host = os.getenv("CORTEX_TF_SERVING_HOST", "localhost")
            cron = TFSModelLoader(
                interval=10,
                model_server_config=model_server_config,
                address=f"{tf_serving_host}:{tf_serving_port}",
                tfs_model_dir=model_dir,
                download_dir=model_dir,
            )
            cron.start()

    # to syncronize with the other serving processes
    open("/run/workspace/init_script_run.txt", "a").close()

    # don't exit the script if the cron is running
    while cron and cron.is_alive():
        time.sleep(0.25)

    # exit if cron has exited with errors
    if cron and isinstance(cron.exitcode, int) and cron.exitcode != 0:
        # if it was killed by a catchable signal
        if cron.exitcode < 0:
            sys.exit(-cron.exitcode)
        sys.exit(cron.exitcode)


if __name__ == "__main__":
    main()
