"""Encode + decode the unbiased eval set (data_eval/) at 5 QPs."""

from pathlib import Path

from decoder.config import BASE_CONFIG as DEC_BASE_CONFIG
from decoder.decoders import VTMDecoder
from decoder.manager import DecoderManager
from encoder.config import BASE_CONFIG as ENC_BASE_CONFIG
from encoder.encoders import VVencEncoder
from encoder.manager import EncoderManager


DATA_DIR = "data_eval"
ENC_OUT = "output_eval/encoded"
DEC_OUT = "output_eval/decoded"
QPS = [22, 27, 32, 37, 42]
FRAMES = 64
WORKERS = 4


def encode():
    ENC_BASE_CONFIG.data_dir = DATA_DIR
    ENC_BASE_CONFIG.output_dir = ENC_OUT
    ENC_BASE_CONFIG.qp = QPS
    ENC_BASE_CONFIG.frames_to_encode = FRAMES
    ENC_BASE_CONFIG.preset = "fast"
    ENC_BASE_CONFIG.alf = 0
    ENC_BASE_CONFIG.sao = 0
    ENC_BASE_CONFIG.max_workers = WORKERS

    mgr = EncoderManager(ENC_BASE_CONFIG, VVencEncoder())
    return mgr.run()


def decode(bitstreams):
    DEC_BASE_CONFIG.bitstream_input = [str(b) for b in bitstreams]
    DEC_BASE_CONFIG.output_path = DEC_OUT
    DEC_BASE_CONFIG.max_workers = WORKERS

    mgr = DecoderManager(DEC_BASE_CONFIG, VTMDecoder())
    return mgr.run()


if __name__ == "__main__":
    yuvs = sorted(Path(DATA_DIR).glob("*.yuv"))
    print(f"Found {len(yuvs)} YUVs in {DATA_DIR}: {[p.name for p in yuvs]}")

    print(f"\nEncoding at QPs {QPS}...")
    bitstreams = encode()
    print(f"\nGot {len(bitstreams)} bitstreams")

    # decode all bitstreams in one batch
    print(f"\nDecoding all bitstreams with VTM...")
    decode(bitstreams)
    print("\nDone.")
