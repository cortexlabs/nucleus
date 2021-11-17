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
