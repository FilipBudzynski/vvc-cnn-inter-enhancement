import yaml
from encoder.config import BASE_CONFIG
from encoder.encoders import VVencEncoder
from encoder.manager import EncoderManager


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vvenc = VVencEncoder(executable_path=config["paths"]["encoder_path"])
    manager = EncoderManager(BASE_CONFIG, vvenc)
    manager.run()
