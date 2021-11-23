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

import os
from typing import List

import pkgutil
import pathlib

import yaml
import click


class CortexModelServerBuilder(RuntimeError):
    pass


def read_model_server_config(file) -> dict:
    """
    Read the cortex model server config file.

    :param file: The YAML file containing the model server config.
    :return: The model server config.
    """
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data


def validate_config(config: dict):
    if config.get("type") not in ["python", "tensorflow"]:
        raise CortexModelServerBuilder("'type' must be set to 'python' or 'tensorflow'")

    if "py_version" not in config:
        config["py_version"] = "3.6.9"

    if "use_local_cortex_libs" not in config:
        config["use_local_cortex_libs"] = False

    if "serve_port" not in config:
        config["serve_port"] = 8080
    if "processes" not in config:
        config["processes"] = 1
    if "threads_per_process" not in config:
        config["threads_per_process"] = 1
    if "max_concurrency" not in config:
        config["max_concurrency"] = 0

    if "dependencies" not in config:
        config["dependencies"] = {
            "pip": "requirements.txt",
            "conda": "conda-packages.txt",
            "shell": "dependencies.sh",
        }
    elif "pip" not in config["dependencies"]:
        config["dependencies"]["pip"] = "requirements.txt"
    elif "conda" not in config["dependencies"]:
        config["dependencies"]["conda"] = "conda-packages.txt"
    elif "shell" not in config["dependencies"]:
        config["dependencies"]["shell"] = "dependencies.sh"

    if "server_side_batching" in config:
        if "max_batch_size" not in config["server_side_batching"]:
            raise CortexModelServerBuilder("server_side_batching: missing 'max_batch_size' field")
        if "batch_interval_seconds" not in config["server_side_batching"]:
            raise CortexModelServerBuilder(
                "server_side_batching: missing 'batch_interval_seconds' field"
            )
        if not isinstance(config["server_side_batching"]["max_batch_size"], int):
            raise CortexModelServerBuilder(
                "server_side_batching: 'max_batch_size' field isn't of type 'int'"
            )
        if not isinstance(config["server_side_batching"]["batch_interval_seconds"], float):
            raise CortexModelServerBuilder(
                "server_side_batching: 'batch_interval_seconds' field isn't of type 'float'"
            )

    if "path" not in config:
        raise CortexModelServerBuilder("'path' field missing")

    if config["type"] == "python" and "models" in config:
        raise CortexModelServerBuilder("'models' field not supported for 'python' type")
    if config["type"] == "tensorflow" and "multi_model_reloading" in config:
        raise CortexModelServerBuilder(
            "'multi_model_reloading' field not supported for 'tensorflow' type"
        )

    if config["type"] == "python":
        if "tfs_version" in config:
            raise CortexModelServerBuilder("'tfs_version' field not supported for 'python' type")
        if "tfs_container_dns" in config:
            raise CortexModelServerBuilder(
                "'tfs_container_dns' field not supported for 'python' type"
            )

    if config["type"] == "tensorflow":
        if "tfs_version" not in config:
            config["tfs_version"] = "2.3.0"
        if "tfs_container_dns" not in config:
            config["tfs_container_dns"] = "localhost"

    models_fieldname: str
    if config["type"] == "python":
        models_fieldname = "multi_model_reloading"
    if config["type"] == "tensorflow":
        models_fieldname = "models"

    if models_fieldname in config:
        if (
            ("path" in config and ("paths" in config or "dir" in config))
            or ("paths" in config and ("path" in config or "dir" in config))
            or ("dir" in config and ("paths" in config or "path" in config))
        ):
            raise CortexModelServerBuilder(
                f"{models_fieldname}: can only specify 'path', 'paths' or 'dir'"
            )

        if (
            "cache_size" in config[models_fieldname]
            and "disk_cache_size" not in config[models_fieldname]
        ) or (
            "cache_size" not in config[models_fieldname]
            and "disk_cache_size" in config[models_fieldname]
        ):
            raise CortexModelServerBuilder(
                f"{models_fieldname}: when the cache is configured, both 'cache_size' and 'disk_cache_size' fields must be specified"
            )
        if (
            "disk_cache_size" in config[models_fieldname]
            and config[models_fieldname]["cache_size"] > config[models_fieldname]["disk_cache_size"]
        ):
            raise CortexModelServerBuilder(
                f"{models_fieldname}: when the cache is configured, 'cache_size' cannot be larger than 'disk_cache_size'"
            )

        if "paths" in config[models_fieldname]:
            if len(config[models_fieldname]["paths"]) == 0:
                raise CortexModelServerBuilder(
                    f"{models_fieldname}: if the 'path' field list is specified, then its length must be at least 1"
                )
            for idx, path in enumerate(config[models_fieldname]["paths"]):
                if "name" not in path:
                    raise CortexModelServerBuilder(
                        f"{models_fieldname}: paths: {idx}: name field required"
                    )
                if "path" not in path:
                    raise CortexModelServerBuilder(
                        f"{models_fieldname}: paths: {idx}: path field required"
                    )

    if "gpu_version" not in config:
        config["gpu_version"] = None
    if "gpu" not in config:
        config["gpu"] = False

    if config["type"] == "python":
        if not config["gpu"] and config["gpu_version"]:
            raise CortexModelServerBuilder(
                "gpu must be enabled (by setting 'gpu: true') in order to specify 'gpu_version' field"
            )
        if config["gpu"] and config["gpu_version"] is None:
            raise CortexModelServerBuilder(
                "when gpu is enabled, the 'gpu_version' field must be specified too"
            )
    if config["type"] == "tensorflow" and config["gpu_version"]:
        raise CortexModelServerBuilder("'gpu_version' field not supported for 'tensorflow' type")

    if config["gpu_version"]:
        if "cuda" not in config["gpu_version"]:
            raise CortexModelServerBuilder("gpu_version: 'cuda' field must be specified")
        if "cudnn" not in config["gpu_version"]:
            raise CortexModelServerBuilder("gpu_version: 'cudnn' field must be specified")


def build_handler_dockerfile(config: dict, path_to_config: str, dev_env: bool) -> str:
    handler_template = pkgutil.get_data(__name__, "templates/handler.Dockerfile")
    base_image = "ubuntu:18.04"
    cortex_image_type = "python-handler-cpu"
    if os.getenv("CORTEX_TELEMETRY_SENTRY_DSN"):
        cortex_sentry_dsn = os.environ["CORTEX_TELEMETRY_SENTRY_DSN"]
    else:
        cortex_sentry_dsn = ""

    if (
        config["type"] == "python"
        and config["gpu_version"]
        and config["gpu_version"]["cuda"] not in ["", None]
        and config["gpu_version"]["cudnn"] not in ["", None]
    ):
        base_image = f"nvidia/cuda:{config['gpu_version']['cuda']}-cudnn{config['gpu_version']['cudnn']}-runtime-ubuntu18.04"
        cortex_image_type = "python-handler-gpu"
    if config["type"] == "tensorflow":
        cortex_image_type = "tensorflow-handler"

    tfs_container_dns = "" if "tfs_container_dns" not in config else config["tfs_container_dns"]
    substitute_envs = {
        "BASE_IMAGE": base_image,
        "PYTHON_VERSION": config["py_version"],
        "CORTEX_IMAGE_TYPE": cortex_image_type,
        "CORTEX_TF_SERVING_HOST": tfs_container_dns,
    }
    for env, val in substitute_envs.items():
        env_var = f"${env}"
        handler_template = handler_template.replace(env_var.encode("utf-8"), val.encode("utf-8"))
    handler_lines = handler_template.splitlines()
    handler_lines = [line.decode("utf-8") for line in handler_lines]

    # create handler dockerfile
    if dev_env:
        handler_lines += [
            "COPY src/cortex/serve.requirements.txt /src/cortex/serve.requirements.txt",
            "COPY src/cortex/cortex_internal.requirements.txt /src/cortex/cortex_internal.requirements.txt",
            "",
        ]
    else:
        handler_lines += [
            "RUN git clone --depth 1 -b ${CORTEX_MODEL_SERVER_VERSION} https://github.com/cortexlabs/nucleus && \\",
            "    cp -r nucleus/src/ /src/ && \\",
            "    rm -r nucleus/",
            "",
        ]

    project_dir = pathlib.Path(path_to_config).parent.relative_to(".")
    config_filename = pathlib.Path(path_to_config).name
    handler_lines += [
        "RUN pip install --no-cache-dir \\",
        "    -r /src/cortex/serve.requirements.txt \\",
        "    -r /src/cortex/cortex_internal.requirements.txt",
    ]
    if dev_env:
        handler_lines += ["", "COPY src/ /src/"]

    env_lines = []
    if cortex_sentry_dsn != "":
        env_lines += ["ENV CORTEX_LOG_CONFIG_FILE=/src/cortex/log_config.yaml \\"]
        if cortex_sentry_dsn != "" and config["type"] == "python":
            env_lines += [
                f'    CORTEX_TELEMETRY_SENTRY_DSN="{cortex_sentry_dsn}"',
            ]
        elif cortex_sentry_dsn != "" and config["type"] == "tensorflow":
            env_lines += [
                f'    CORTEX_TELEMETRY_SENTRY_DSN="{cortex_sentry_dsn}" \\',
                f"    CORTEX_TFS_VERSION={config['tfs_version']}",
            ]
    else:
        if config["type"] == "python":
            env_lines += ["ENV CORTEX_LOG_CONFIG_FILE=/src/cortex/log_config.yaml"]
        else:
            env_lines += [
                "ENV CORTEX_LOG_CONFIG_FILE=/src/cortex/log_config.yaml \\",
                f"    CORTEX_TFS_VERSION={config['tfs_version']}",
            ]

    handler_lines += [
        *env_lines,
        "",
        "RUN mkdir -p /usr/local/cortex/ && \\",
        "    cp /src/cortex/init/install-core-dependencies.sh /usr/local/cortex/install-core-dependencies.sh && \\",
        "    chmod +x /usr/local/cortex/install-core-dependencies.sh && \\",
        "    /usr/local/cortex/install-core-dependencies.sh",
        "",
        "RUN pip install --no-deps /src/cortex/ && \\",
        "    mv /src/cortex/init/bootloader.sh /etc/cont-init.d/bootloader.sh",
        "",
    ]

    env_file_exists = (project_dir / ".env").exists()
    pip_deps_exists = (project_dir / config["dependencies"]["pip"]).exists()
    conda_deps_exists = (project_dir / config["dependencies"]["conda"]).exists()
    shell_deps_exists = (project_dir / config["dependencies"]["shell"]).exists()

    if env_file_exists:
        handler_lines += [f"COPY {project_dir}/.env /src/project/"]
    if shell_deps_exists:
        handler_lines += [f"COPY {project_dir}/{config['dependencies']['shell']} /src/project/"]
    if conda_deps_exists:
        handler_lines += [f"COPY {project_dir}/{config['dependencies']['conda']} /src/project/"]
    if pip_deps_exists:
        handler_lines += [f"COPY {project_dir}/{config['dependencies']['pip']} /src/project/"]
    handler_lines += [
        f"COPY {project_dir}/{config_filename} /src/project/{config_filename}",
        "",
        f"ENV CORTEX_MODEL_SERVER_CONFIG /src/project/{config_filename}",
        f"RUN /opt/conda/envs/env/bin/python /src/cortex/init/expand_server_config.py /src/project/{config_filename} > /tmp/{config_filename} && \\",
        f"   eval $(/opt/conda/envs/env/bin/python /src/cortex/init/export_env_vars.py /tmp/{config_filename}) && \\",
    ]

    if env_file_exists:
        handler_lines += [
            f'   if [ -f "/src/project/.env" ]; then set -a; source /src/project/.env; set +a; fi && \\',
        ]
    if shell_deps_exists:
        handler_lines += [
            '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_SHELL}" ]; then bash -e "/src/project/${CORTEX_DEPENDENCIES_SHELL}"; fi && \\',
        ]
    if conda_deps_exists:
        handler_lines += [
            '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_CONDA}" ]; then conda config --append channels conda-forge && conda install -y --file "/src/project/${CORTEX_DEPENDENCIES_CONDA}"; fi && \\',
        ]
    if pip_deps_exists:
        handler_lines += [
            '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_PIP}" ]; then pip --no-cache-dir install -r "/src/project/${CORTEX_DEPENDENCIES_PIP}"; fi && \\',
        ]

    handler_lines += [
        "    /usr/local/cortex/install-core-dependencies.sh",
        "",
        f"COPY {project_dir}/ /src/project/",
        f"RUN mv /tmp/{config_filename} /src/project/{config_filename}",
        "",
        'ENTRYPOINT ["/init"]',
        "",
    ]

    handler_dockerfile = "\n".join(handler_lines)
    return handler_dockerfile


def build_tensorflow_dockerfile(config: dict, tfs_dockerfile: bytes, dev_env: bool) -> str:
    if config["gpu"]:
        tfs_dockerfile = tfs_dockerfile.replace(
            "$BASE_IMAGE".encode("utf-8"),
            f"tensorflow/serving:{config['tfs_version']}-gpu".encode("utf-8"),
        )
    else:
        tfs_dockerfile = tfs_dockerfile.replace(
            "$BASE_IMAGE".encode("utf-8"),
            f"tensorflow/serving:{config['tfs_version']}".encode("utf-8"),
        )
    tfs_template_lines = tfs_dockerfile.splitlines()
    tfs_lines = [line.decode("utf-8") for line in tfs_template_lines]

    tfs_lines += [
        "",
        "RUN apt-get update -qq && apt-get install -y --no-install-recommends -q \\",
        "    curl \\",
        "    git && \\",
        "    apt-get clean -qq && rm -rf /var/lib/apt/lists/*",
        "",
    ]

    tf_base_serving_port = 9000
    tf_empty_model_config = "/etc/tfs/model_config_server.conf"
    tf_max_num_load_retries = "0"
    tf_load_time_micros = "30000000"
    grpc_max_concurrent_streams = int(config["processes"]) * int(config["threads_per_process"]) + 10
    batch_params_file = "/etc/tfs/batch_config.conf"

    if dev_env:
        tfs_lines += [
            "COPY nucleus/templates/tfs-run.sh /src/",
        ]
    else:
        tfs_lines += [
            "RUN git clone --depth 1 -b ${CORTEX_MODEL_SERVER_VERSION} https://github.com/cortexlabs/nucleus \\",
            "    cp nucleus/nucleus/templates/tfs-run.sh /src/ && \\",
            "    rm -r nucleus/",
        ]
    tfs_lines += [
        "RUN chmod +x /src/tfs-run.sh",
        "",
    ]
    if (
        "server_side_batching" in config
        and config["server_side_batching"]["max_batch_size"]
        and config["server_side_batching"]["batch_interval_seconds"]
    ):
        tfs_lines += [
            f"ENV TF_MAX_BATCH_SIZE={config['server_side_batching']['max_batch_size']} \\",
            f"    TF_BATCH_TIMEOUT_MICROS={int(float(config['server_side_batching']['batch_interval_seconds']) * 1000000)} \\",
            f"    TF_NUM_BATCHED_THREADS={config['processes']}",
            "",
            f'ENTRYPOINT ["/src/tfs-run.sh", "--port={tf_base_serving_port}", "--model_config_file={tf_empty_model_config}", "--max_num_load_retries={tf_max_num_load_retries}", "--load_retry_interval_micros={tf_load_time_micros}", "--grpc_channel_arguments=\'grpc.max_concurrent_streams={grpc_max_concurrent_streams}\'", "--enable_batching=true", "--batching_parameters_file={batch_params_file}"]',
        ]
    else:
        tfs_lines += [
            f'ENTRYPOINT ["/src/tfs-run.sh", "--port={tf_base_serving_port}", "--model_config_file={tf_empty_model_config}", "--max_num_load_retries={tf_max_num_load_retries}", "--load_retry_interval_micros={tf_load_time_micros}", "--grpc_channel_arguments=\'grpc.max_concurrent_streams={grpc_max_concurrent_streams}\'"]'
        ]
    tfs_lines.append("")

    tensorflow_dockerfile = "\n".join(tfs_lines)
    return tensorflow_dockerfile


def build_dockerfile_images(config: dict, path_to_config: str) -> List[str]:
    validate_config(config)

    nucleus_file = "nucleus.Dockerfile"
    nucleus_tfs_file = "nucleus-tfs.Dockerfile"

    click.echo("-------------- nucleus model server config --------------")
    click.echo(yaml.dump(config, indent=2, sort_keys=False))
    click.echo("---------------------------------------------------------")

    # if using the local files or the git clone in the Dockerfiles
    is_dev_env = config["use_local_cortex_libs"]

    # get handler template
    click.echo(f"generating {nucleus_file}...")
    handler_dockerfile = build_handler_dockerfile(config, path_to_config, is_dev_env)
    with open(nucleus_file, "w") as f:
        f.write(handler_dockerfile)
    click.echo(f"✓ done")

    # get tfs template
    if config["type"] == "tensorflow":
        click.echo(f"generating {nucleus_tfs_file}...")
        tfs_dockerfile = pkgutil.get_data(__name__, "templates/tfs.Dockerfile")
        tensorflow_dockerfile = build_tensorflow_dockerfile(config, tfs_dockerfile, is_dev_env)
        with open(nucleus_tfs_file, "w") as f:
            f.write(tensorflow_dockerfile)
        click.echo(f"✓ done")


@click.command(
    help="A utility to generate Dockerfile(s) for Nucleus model servers. Compatible with Cortex clusters."
)
@click.argument("config", required=True, type=str, default="nucleus-model-server-config.yaml")
def generate(config):
    # get the model server config
    server_config = read_model_server_config(config)
    build_dockerfile_images(server_config, config)
