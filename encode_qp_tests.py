#!/usr/bin/env python3
"""
Koduje wideo dla różnych QP i przygotowuje dataset do ewaluacji
"""

import yaml
from encoder.config import Config
from encoder.encoders import VVencEncoder
from encoder.manager import EncoderManager

QP_VALUES = [22, 27, 32, 37, 42]

def main():
    for qp in QP_VALUES:
        print(f"\n{'='*60}")
        print(f"Kodowanie dla QP = {qp}")
        print(f"{'='*60}")
        
        # Konfiguracja
        cfg = Config()
        cfg.qp = [qp]
        cfg.data_dir = "./data"
        cfg.output_dir = f"./output_qp{qp}/encoded"
        cfg.frames_to_encode = 64  # Ile ramek zakodować
        cfg.alf = 0
        cfg.sao = 0
        
        # Kodowanie
        encoder = VVencEncoder()
        manager = EncoderManager(cfg, encoder)
        results = manager.run()
        
        print(f"Zakodowano {len(results)} plików dla QP={qp}")

if __name__ == "__main__":
    main()
