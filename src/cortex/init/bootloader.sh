#!/usr/bin/with-contenv bash

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

set -e

eval $(/opt/conda/envs/env/bin/python /src/cortex/init/export_env_vars.py $CORTEX_MODEL_SERVER_CONFIG)

# print the model server config
echo "------------ cortex model server config ------------"
cat $CORTEX_MODEL_SERVER_CONFIG
echo "----------------------------------------------------"

function substitute_env_vars() {
    file_to_run_substitution=$1
    /opt/conda/envs/env/bin/python -c "from cortex_internal.lib import util; import os; util.expand_environment_vars_on_file('$file_to_run_substitution')"
}

# configure log level for python scripts§
substitute_env_vars $CORTEX_LOG_CONFIG_FILE

mkdir -p /run/workspace
mkdir -p /run/requests

cd /src/project

# if the container restarted, ensure that it is not perceived as ready
rm -rf /run/workspace/api_readiness.txt
rm -rf /run/workspace/init_script_run.txt
rm -rf /run/workspace/proc-*-ready.txt

sysctl -w net.core.somaxconn="65535" >/dev/null
sysctl -w net.ipv4.ip_local_port_range="15000 64000" >/dev/null
sysctl -w net.ipv4.tcp_fin_timeout=30 >/dev/null

# to export user-specified environment files
source_env_file_cmd="if [ -f \"/src/project/.env\" ]; then set -a; source /src/project/.env; set +a; fi"

# only terminate pod if this process exits with non-zero exit code
create_s6_service() {
    export SERVICE_NAME=$1
    export COMMAND_TO_RUN=$2

    dest_dir="/etc/services.d/$SERVICE_NAME"
    mkdir $dest_dir

    dest_script="$dest_dir/run"
    cp /src/cortex/init/templates/run $dest_script
    substitute_env_vars $dest_script
    chmod +x $dest_script

    dest_script="$dest_dir/finish"
    cp /src/cortex/init/templates/finish $dest_script
    substitute_env_vars $dest_script
    chmod +x $dest_script

    unset SERVICE_NAME
    unset COMMAND_TO_RUN
}

# only terminate pod if this process exits with non-zero exit code
create_s6_service_from_file() {
    export SERVICE_NAME=$1
    runnable=$2

    dest_dir="/etc/services.d/$SERVICE_NAME"
    mkdir $dest_dir

    cp $runnable $dest_dir/run
    chmod +x $dest_dir/run

    dest_script="$dest_dir/finish"
    cp /src/cortex/init/templates/finish $dest_script
    substitute_env_vars $dest_script
    chmod +x $dest_script

    unset SERVICE_NAME
}

# prepare webserver
if [ $CORTEX_SERVING_PROTOCOL = "http" ]; then
    mkdir /run/servers
fi

if [ $CORTEX_SERVING_PROTOCOL = "grpc" ]; then
    /opt/conda/envs/env/bin/python -m grpc_tools.protoc --proto_path=$CORTEX_PROJECT_DIR --python_out=$CORTEX_PYTHON_PATH --grpc_python_out=$CORTEX_PYTHON_PATH $CORTEX_PROTOBUF_FILE
fi

# prepare servers
for i in $(seq 1 $CORTEX_PROCESSES); do
    # prepare uvicorn workers
    if [ $CORTEX_SERVING_PROTOCOL = "http" ]; then
        create_s6_service "uvicorn-$((i-1))" "cd /src/project && $source_env_file_cmd && PYTHONUNBUFFERED=TRUE PYTHONPATH=$PYTHONPATH:$CORTEX_PYTHON_PATH exec /opt/conda/envs/env/bin/python /src/cortex/start/server.py /run/servers/proc-$((i-1)).sock"
    fi

    # prepare grpc workers
    if [ $CORTEX_SERVING_PROTOCOL = "grpc" ]; then
        create_s6_service "grpc-$((i-1))" "cd /src/project && $source_env_file_cmd && PYTHONUNBUFFERED=TRUE PYTHONPATH=$PYTHONPATH:$CORTEX_PYTHON_PATH exec /opt/conda/envs/env/bin/python /src/cortex/start/server_grpc.py localhost:$((i-1+20000))"
    fi
done

# generate nginx conf
/opt/conda/envs/env/bin/python -c 'from cortex_internal.lib import util; import os; generated = util.render_jinja_template("/src/cortex/nginx.conf.j2", os.environ); print(generated);' > /run/nginx.conf

create_s6_service "py_init" "cd /src/project && exec /opt/conda/envs/env/bin/python /src/cortex/init/script.py"
create_s6_service "nginx" "exec nginx -c /run/nginx.conf"
create_s6_service_from_file "api_readiness" "/src/cortex/poll/readiness.sh"
