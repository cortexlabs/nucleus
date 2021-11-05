from typing import List

import pkgutil
import pathlib

import yaml
import click


def read_model_server_config(file) -> dict:
    """
    Read the cortex model server config file.

    :param file: The YAML file containing the model server config.
    :return: The model server config.
    """
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data


def build_dockerfile_images(config: dict, path_to_config: str) -> List[str]:
    # if using the local files or the git clone in the Dockerfiles
    dev_env = "dev" in config and config["dev"]

    # get handler template
    handler_template = pkgutil.get_data(__name__, "templates/handler.Dockerfile")
    base_image = "ubuntu:18.04"
    cortex_image_type = "python-handler-cpu"
    if (
        config["type"] == "python"
        and config["gpu"]["cuda"] not in ["", None]
        and config["gpu"]["cudnn"] not in ["", None]
    ):
        base_image = (
            f"nvidia/cuda:{config['gpu']['cuda']}-cudnn{config['gpu']['cudnn']}-runtime-ubuntu18.04"
        )
        cortex_image_type = "python-handler-gpu"
    if config["type"] == "tensorflow":
        cortex_image_type = "tensorflow-handler"
    substitute_envs = {
        "BASE_IMAGE": base_image,
        "PYTHON_VERSION": config["py_version"],
        "CORTEX_IMAGE_TYPE": cortex_image_type,
    }
    for env, val in substitute_envs.items():
        env_var = f"${env}"
        handler_template = handler_template.replace(env_var.encode("utf-8"), val.encode("utf-8"))
    handler_lines = handler_template.splitlines()
    handler_lines = [line.decode("utf-8") for line in handler_lines]

    # create handler dockerfile
    if dev_env:
        handler_lines += ["COPY src/ /src/", ""]
    else:
        handler_lines += [
            "RUN git clone --depth 1 https://github.com/robertlucian/cortex-templates && \\",
            "    cp -r cortex-templates/src/* /src/ && \\",
            "    rm -r cortex-templates/",
            "",
        ]

    project_dir = pathlib.Path(path_to_config).parent.absolute()
    config_filename = pathlib.Path(path_to_config).name
    handler_lines += [
        "RUN pip install --no-cache-dir \\",
        "    -r /src/cortex/serve/serve.requirements.txt \\",
        "    -r /src/cortex/serve/cortex_internal.requirements.txt",
        "RUN cp /src/cortex/serve/init/install-core-dependencies.sh /usr/local/cortex/install-core-dependencies.sh && \\",
        "    /usr/local/cortex/install-core-dependencies.sh",
        "ENV CORTEX_LOG_CONFIG_FILE /src/cortex/serve/log_config.yaml",
        "",
        "RUN pip install --no-deps /src/cortex/serve/ && \\",
        "    mv /src/cortex/serve/init/bootloader.sh /etc/cont-init.d/bootloader.sh",
        "",
        f"COPY {project_dir}/ /mnt/project/",
        f"ENV CORTEX_MODEL_SERVER_CONFIG /mnt/project/{config_filename}" "",
        f"RUN eval $(/opt/conda/envs/env/bin/python /src/cortex/serve/init/export_env_vars.py /mnt/project/{config_filename} && \\",
        f'    if [ -f "/mnt/project/.env" ]; then set -a; source /mnt/project/.env; set +a; fi && \\',
        '    if [ -f "/mnt/project/${CORTEX_DEPENDENCIES_SHELL}" ]; then bash -e "/mnt/project/${CORTEX_DEPENDENCIES_SHELL}"; fi && \\',
        '    if [ -f "/mnt/project/${CORTEX_DEPENDENCIES_CONDA}" ]; then conda config --append channels conda-forge && conda install -y --file "/mnt/project/${CORTEX_DEPENDENCIES_CONDA}"; fi && \\',
        '    if [ -f "/mnt/project/${CORTEX_DEPENDENCIES_PIP}" ]; then pip --no-cache-dir install -r "/mnt/project/${CORTEX_DEPENDENCIES_PIP}"; fi && \\',
        "    /usr/local/cortex/install-core-dependencies.sh",
        "",
        'ENTRYPOINT ["/init"]',
    ]

    # get tfs template
    tfs_dockerfile = None
    if (
        config["type"] == "tensorflow"
        and config["gpu"]["cuda"] not in ["", None]
        and config["gpu"]["cudnn"] not in ["", None]
    ):
        tfs_dockerfile = pkgutil.get_data(__name__, "templates/tfs-gpu.Dockerfile")
        tfs_lines = tfs_dockerfile.splitlines()
    elif config["type"] == "tensorflow":
        tfs_dockerfile = pkgutil.get_data(__name__, "templates/tfs-cpu.Dockerfile")
        tfs_lines = tfs_dockerfile.splitlines()

    # create tfs dockerfile
    if tfs_dockerfile:
        tfs_lines = [line.decode("utf-8") for line in tfs_lines]

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
                "RUN git clone --depth 1 https://github.com/robertlucian/cortex-templates \\",
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
                f"ENV TF_MAX_BATCH_SIZE={config['server_side_batching']['max_batch_size']}",
                f"    TF_BATCH_TIMEOUT_MICROS={int(float(config['server_side_batching']['batch_interval']) * 1000000)}",
                f"    TF_NUM_BATCHED_THREADS={config['processes_per_replica']}",
                "",
                f"ENTRYPOINT ['/src/run.sh', '--port={tf_base_serving_port}', '--model_config_file={tf_empty_model_config}', '--max_num_load_retries={tf_max_num_load_retries}', '--load_retry_interval_micros={tf_load_time_micros}', '--grpc_channel_arguments=\"grpc.max_concurrent_streams={grpc_channel_arguments}\"', '--enable_batching=true', '--batching_parameters_file={batch_params_file}']",
            ]
        else:
            tfs_lines += [
                f"ENTRYPOINT ['/src/run.sh', '--port={tf_base_serving_port}', '--model_config_file={tf_empty_model_config}', '--max_num_load_retries={tf_max_num_load_retries}', '--load_retry_interval_micros={tf_load_time_micros}', '--grpc_channel_arguments=\"grpc.max_concurrent_streams={grpc_channel_arguments}\"']"
            ]
        # click.echo("\n".join(tfs_lines))
    click.echo("\n".join(handler_lines))


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
    build_dockerfile_images(config, path_to_config)


if __name__ == "__main__":
    main()
