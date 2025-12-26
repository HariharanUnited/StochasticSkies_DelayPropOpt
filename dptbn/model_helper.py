
class ModelWrapper:
    def __init__(self, model):
        # Unwrap if already wrapped
        if hasattr(model, "model"):
            self.model = model.model
        else:
            self.model = model

def get_model(name: str):
    if name not in MODELS:
        return None
    return ModelWrapper(MODELS[name])
