from lightning.pytorch.cli import LightningCLI

import sys
from pathlib import Path

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.init_args.sequence_length', 'model.init_args.sequence_length')
        parser.link_arguments(
            'data.init_args.features',
            'model.init_args.input_dim',
            compute_fn=len,
        )

def cli_main():
    cli = MyLightningCLI(
        parser_kwargs={'parser_mode': 'omegaconf'},
        save_config_kwargs={'overwrite': True},
    )


if __name__ == "__main__":
    cli_main()