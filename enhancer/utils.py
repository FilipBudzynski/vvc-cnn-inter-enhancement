import torch


def weights_init(model) -> None:
    """weights_init.
    :param model:
    :rtype: None
    """
    classname = model.__class__.__name__

    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.bias, 0)