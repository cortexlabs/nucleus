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

import click
from generate import generate


@click.group(
    help="Use the Nucleus CLI to generate model server for Python-generic and TensorFlow models. Compatible with Cortex clusters."
)
def cli():
    pass


cli.add_command(generate)

if __name__ == "__main__":
    cli()
