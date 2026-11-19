from pathlib import Path
import argparse
import gc
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'working/desktop'))
from live_detector import Detector, ModelChoice, WEIGHTS


def main():
    parser = argparse.ArgumentParser(description='Load and verify locally installed model weights.')
    parser.add_argument('--model', choices=tuple(WEIGHTS), action='append',
                        help='Select one model. Repeat to test multiple models. Defaults to all models.')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    frames = [np.zeros((180, 320, 3), np.uint8) for _ in range(3)]
    results = []
    for name in args.model or WEIGHTS:
        detector = Detector(ModelChoice.preset(name), device)
        prediction = detector.predict(frames, realtime=True)
        assert prediction.shape == (3, 288, 512)
        assert np.isfinite(prediction).all()
        results.append({'model': name, 'device': str(device), 'shape': list(prediction.shape), 'finite': True})
        print('Loaded and inferred:', name, flush=True)
        del detector, prediction
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    destination = ROOT / 'test_artifacts'
    destination.mkdir(exist_ok=True)
    (destination / 'model-smoke.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
