import yaml

from encoder.encoders import VVencEncoder
from encoder.manager import EncoderManager

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vvenc = VVencEncoder(executable_path=config["paths"]["encoder_path"])
    manager = EncoderManager(config, vvenc)
    manager.run()
