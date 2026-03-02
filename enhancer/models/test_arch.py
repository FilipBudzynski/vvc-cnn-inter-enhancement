import torch
from enhancer.models.dense import DenseNet
from enhancer.config import EnhancerConfig, NetworkImplementation
from enhancer.models.res import ResNet


def test_models():
    config_dict = {
        "input_shape": [10, 128, 128],
        "implementation": "res",  # Testing ResNet now
        "features": {"features": 32, "res": True},
        "structure": {"blocks": [{"num_layers": 3, "features": 32}]},
        "output_block": {"features": 3},
    }
    config = EnhancerConfig(**config_dict)

    # Test ResNet
    print("Testing ResNet...")
    res_model = ResNet(config)
    dummy_in = torch.randn(1, 10, 128, 128)
    res_out = res_model(dummy_in)
    print(f"ResNet Output: {res_out.shape} ✅")

    # Test DenseNet
    print("\nTesting DenseNet...")
    config.implementation = NetworkImplementation.DENSE
    dense_model = DenseNet(config)
    dense_out = dense_model(dummy_in)
    print(f"DenseNet Output: {dense_out.shape} ✅")


if __name__ == "__main__":
    test_models()
