import pydoc

import yaml


def load_model(exp_dir, ckpt_type):
    with open(exp_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_class_name = config['model']['class_path']
    model_class = pydoc.locate(model_class_name)
    model = model_class.load_from_checkpoint(
        exp_dir / f'checkpoints/{ckpt_type}.ckpt',
    )
    return model
