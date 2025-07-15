import torch


def get_full_optimizer_type_name(optimizer: torch.optim.Optimizer) -> str:
    """
    Get the full type name of the optimizer, including the module path.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.

    Returns:
        str: The full type name of the optimizer.
    """
    return f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}"
