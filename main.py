import yaml
from decoder.decoders import VTMDecoder
from decoder.manager import DecoderManager
from encoder.config import BASE_CONFIG as ENC_BASE_CONFIG
from decoder.config import BASE_CONFIG as DEC_BASE_CONFIG
from encoder.encoders import VVencEncoder
from encoder.manager import EncoderManager


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vvenc = VVencEncoder()
    enc_mgr = EncoderManager(ENC_BASE_CONFIG, vvenc)

    bitstream_path = enc_mgr.run()

    dec_config = DEC_BASE_CONFIG.bitstream_path = bitstream_path
    vtm_dec = VTMDecoder()
    dec_mgr = DecoderManager(DEC_BASE_CONFIG, decoder=vtm_dec)
    dec_mgr.run()
