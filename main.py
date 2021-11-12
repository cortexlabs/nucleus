import os
from typing import List

import pkgutil
import pathlib

import yaml
import click

CORTEX_MODEL_SERVER_VERSION = "master"

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
    if "type" not in config:
        raise RuntimeError("missing 'type'")

    if "py_version" not in config:
        config["py_version"] = "3.6.9"

    if "use_local_cortex_libs" not in config:
        config["use_local_cortex_libs"] = False

    if "processes_per_replica" not in config:
        config["processes_per_replica"] = 1
    if "threads_per_process" not in config:
        config["threads_per_process"] = 1
    if "max_replica_concurrency" not in config:
        config["max_replica_concurrency"] = 1

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
        if "batch_interval" not in config["server_side_batching"]:
            raise CortexModelServerBuilder("server_side_batching: missing 'batch_interval' field")
        if not isinstance(config["server_side_batching"]["max_batch_size"], int):
            raise CortexModelServerBuilder(
                "server_side_batching: 'max_batch_size' field isn't of type 'int'"
            )
        if not isinstance(config["server_side_batching"]["batch_interval"], float):
            raise CortexModelServerBuilder(
                "server_side_batching: 'batch_interval' field isn't of type 'float'"
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
    elif ("cuda" in config["gpu_version"] and "cudnn" not in config["gpu_version"]) or (
        "cuda" in config["gpu_version"] and "cudnn" not in config["gpu_version"]
    ):
        raise CortexModelServerBuilder(
            "gpu_version: both 'cuda' and 'cudnn' fields must be specified"
        )


def build_handler_dockerfile(config: dict, path_to_config: str, dev_env: bool) -> str:
    handler_template = pkgutil.get_data(__name__, "templates/handler.Dockerfile")
    base_image = "ubuntu:18.04"
    cortex_image_type = "python-handler-cpu"
    if os.environ["CORTEX_TELEMETRY_SENTRY_DSN"]:
        cortex_sentry_dsn = os.getenv("CORTEX_TELEMETRY_SENTRY_DSN")
    else:
        cortex_sentry_dsn = "https://5cea3d2d67194d028f7191fcc6ebca14@sentry.io/1825326"
    cortex_telemetry_sentry_environment = "nucleus"

    if (
        config["type"] == "python"
        and config["gpu_version"]
        and config["gpu_version"]["cuda"] not in ["", None]
        and config["gpu_version"]["cudnn"] not in ["", None]
    ):
        base_image = (
            f"nvidia/cuda:{config['gpu_version']['cuda']}-cudnn{config['gpu_version']['cudnn']}-runtime-ubuntu18.04"
        )
        cortex_image_type = "python-handler-gpu"
    if config["type"] == "tensorflow":
        cortex_image_type = "tensorflow-handler"
    substitute_envs = {
        "BASE_IMAGE": base_image,
        "PYTHON_VERSION": config["py_version"],
        "CORTEX_IMAGE_TYPE": cortex_image_type,
        "CORTEX_TF_SERVING_HOST": "" if "tfs_container_dns" not in config else config["tfs_container_dns"],
    }
    for env, val in substitute_envs.items():
        env_var = f"${env}"
        handler_template = handler_template.replace(env_var.encode("utf-8"), val.encode("utf-8"))
    handler_lines = handler_template.splitlines()
    handler_lines = [line.decode("utf-8") for line in handler_lines]

    # create handler dockerfile
    if dev_env:
        handler_lines += [
            "COPY /src/cortex/serve.requirements.txt /src/cortex/serve.requirements.txt",
            "COPY /src/cortex/cortex_internal.requirements.txt /src/cortex/cortex_internal.requirements.txt",
            "",
        ]
    else:
        handler_lines += [
            "RUN git clone --depth 1 -b v{CORTEX_MODEL_SERVER_VERSION} https://github.com/cortexlabs/nucleus && \\",
            "    cp -r cortex-templates/src/* /src/ && \\",
            "    rm -r cortex-templates/",
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
    handler_lines += [
        "RUN mkdir -p /usr/local/cortex/ && \\",
        "    cp /src/cortex/init/install-core-dependencies.sh /usr/local/cortex/install-core-dependencies.sh && \\",
        "    chmod +x /usr/local/cortex/install-core-dependencies.sh && \\",
        "    /usr/local/cortex/install-core-dependencies.sh",
        "ENV CORTEX_LOG_CONFIG_FILE /src/cortex/log_config.yaml \\",
        f"   CORTEX_TELEMETRY_SENTRY_DSN {cortex_sentry_dsn} \\",
        f"   CORTEX_TELEMETRY_SENTRY_ENVIRONMENT {cortex_telemetry_sentry_environment}",
        "",
        "RUN pip install --no-deps /src/cortex/ && \\",
        "    mv /src/cortex/init/bootloader.sh /etc/cont-init.d/bootloader.sh",
        "",
        f"COPY {project_dir}/ /src/project/",
        f"ENV CORTEX_MODEL_SERVER_CONFIG /src/project/{config_filename}",
        f"RUN /opt/conda/envs/env/bin/python /src/cortex/init/expand_server_config.py /src/project/{config_filename} > /src/project/{config_filename}.tmp && \\",
        f"   mv /src/project/{config_filename}.tmp /src/project/{config_filename} && \\",
        f"   eval $(/opt/conda/envs/env/bin/python /src/cortex/init/export_env_vars.py /src/project/{config_filename}) && \\",
        f'   if [ -f "/src/project/.env" ]; then set -a; source /src/project/.env; set +a; fi && \\',
        '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_SHELL}" ]; then bash -e "/src/project/${CORTEX_DEPENDENCIES_SHELL}"; fi && \\',
        '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_CONDA}" ]; then conda config --append channels conda-forge && conda install -y --file "/src/project/${CORTEX_DEPENDENCIES_CONDA}"; fi && \\',
        '    if [ -f "/src/project/${CORTEX_DEPENDENCIES_PIP}" ]; then pip --no-cache-dir install -r "/src/project/${CORTEX_DEPENDENCIES_PIP}"; fi && \\',
        "    /usr/local/cortex/install-core-dependencies.sh",
        "",
        'ENTRYPOINT ["/init"]',
        "",
    ]

    handler_dockerfile = "\n".join(handler_lines)
    return handler_dockerfile


def build_tensorflow_dockerfile(config: dict, tfs_dockerfile: bytes, dev_env: bool) -> str:
    runs_on_gpu = False
    if (
        config["gpu_version"]
        and config["gpu_version"]["cuda"] not in ["", None]
        and config["gpu_version"]["cudnn"] not in ["", None]
    ):
        tfs_dockerfile = tfs_dockerfile.replace(
            "$BASE_IMAGE".encode("utf-8"),
            f"tensorflow/serving:{config['tfs_version']}-gpu".encode("utf-8"),
        )
        runs_on_gpu = True
    else:
        tfs_dockerfile = tfs_dockerfile.replace(
            "$BASE_IMAGE".encode("utf-8"),
            f"tensorflow/serving:{config['tfs_version']}".encode("utf-8"),
        )
    tfs_template_lines = tfs_dockerfile.splitlines()
    tfs_lines = [line.decode("utf-8") for line in tfs_template_lines]

    if runs_on_gpu:
        tfs_lines += [
            "",
            "RUN apt-get update -qq && apt-get install -y --no-install-recommends -q \\",
            "    libnvinfer6=6.0.1-1+cuda10.1 \\",
            "    libnvinfer-plugin6=6.0.1-1+cuda10.1 \\",
            "    curl \\",
            "    git && \\",
            "    apt-get clean -qq && rm -rf /var/lib/apt/lists/*",
            "",
        ]
    else:
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
    grpc_channel_arguments = (
        int(config["processes_per_replica"]) * int(config["threads_per_process"]) + 10
    )
    batch_params_file = "/etc/tfs/batch_config.conf"

    if dev_env:
        tfs_lines += [
            "COPY templates/tfs-run.sh /src/",
        ]
    else:
        tfs_lines += [
            f"RUN git clone --depth 1 -b v{CORTEX_MODEL_SERVER_VERSION} https://github.com/cortexlabs/nucleus \\",
            "    cp cortex-templates/data/tfs-run.sh /src/ && \\",
            "    rm -r cortex-templates/",
        ]
    tfs_lines += [
        "RUN chmod +x /src/tfs-run.sh",
        "",
    ]
    if (
        "server_side_batching" in config
        and config["server_side_batching"]["max_batch_size"]
        and config["server_side_batching"]["batch_interval"]
    ):
        tfs_lines += [
            f"ENV TF_MAX_BATCH_SIZE={config['server_side_batching']['max_batch_size']} \\",
            f"    TF_BATCH_TIMEOUT_MICROS={int(float(config['server_side_batching']['batch_interval']) * 1000000)} \\",
            f"    TF_NUM_BATCHED_THREADS={config['processes_per_replica']}",
            "",
            f'ENTRYPOINT ["/src/tfs-run.sh", "--port={tf_base_serving_port}", "--model_config_file={tf_empty_model_config}", "--max_num_load_retries={tf_max_num_load_retries}", "--load_retry_interval_micros={tf_load_time_micros}", "--grpc_channel_arguments=\'grpc.max_concurrent_streams={grpc_channel_arguments}\'", "--enable_batching=true", "--batching_parameters_file={batch_params_file}"]',
        ]
    else:
        tfs_lines += [
            f'ENTRYPOINT ["/src/tfs-run.sh", "--port={tf_base_serving_port}", "--model_config_file={tf_empty_model_config}", "--max_num_load_retries={tf_max_num_load_retries}", "--load_retry_interval_micros={tf_load_time_micros}", "--grpc_channel_arguments=\'grpc.max_concurrent_streams={grpc_channel_arguments}\'"]'
        ]
    tfs_lines.append("")

    tensorflow_dockerfile = "\n".join(tfs_lines)
    return tensorflow_dockerfile


def build_dockerfile_images(
    config: dict, path_to_config: str, dockerfile_output_prefix: str
) -> List[str]:
    validate_config(config)

    click.echo("generating dockerfiles for the following model server config")
    click.echo("--------------------------------------------------------->")
    click.echo(yaml.dump(config, indent=2, sort_keys=False))
    click.echo("--------------------------------------------------------->")

    # if using the local files or the git clone in the Dockerfiles
    dev_env = config["use_local_cortex_libs"]

    # get handler template
    click.echo("generating handler dockerfile ...")
    handler_dockerfile = build_handler_dockerfile(config, path_to_config, dev_env)
    click.echo(f"outputting handler handler-{dockerfile_output_prefix}.Dockerfile")
    with open(f"handler-{dockerfile_output_prefix}.Dockerfile", "w") as f:
        f.write(handler_dockerfile)

    # get tfs template
    if config["type"] == "tensorflow":
        click.echo("generating tensorflow dockerfile ...")
        tfs_dockerfile = pkgutil.get_data(__name__, "templates/tfs.Dockerfile")
        tensorflow_dockerfile = build_tensorflow_dockerfile(config, tfs_dockerfile, dev_env)
        click.echo(f"outputting tensorflow tfs-{dockerfile_output_prefix}.Dockerfile")
        with open(f"tfs-{dockerfile_output_prefix}.Dockerfile", "w") as f:
            f.write(tensorflow_dockerfile)


@click.command(help="A Cortex utility to build model servers without caring about dockerfiles")
@click.option(
    "--path-to-config",
    type=click.STRING,
    default="cortex-model-server-config.yaml",
    show_default=True,
    help="Path to model server config; the config's dir represents the target dir project",
)
@click.option(
    "--dockerfile-output-prefix",
    type=click.STRING,
    default="cortex-model-server",
    show_default=True,
    help="The output dockerfile(s)' prefix",
)
def main(path_to_config, dockerfile_output_prefix):
    # get the model server config
    config = read_model_server_config(path_to_config)
    build_dockerfile_images(config, path_to_config, dockerfile_output_prefix)


if __name__ == "__main__":
    main()
