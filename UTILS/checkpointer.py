import torch


def save(model, path):
    if not hasattr(model, "init_args"):
        raise AttributeError("Model must have `init_args` attribute for checkpointing.")
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_args": model.init_args
    }

    torch.save(checkpoint, path)


def load(ModelClass, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_args = checkpoint["model_args"]
    model = ModelClass(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
