import torch.nn as nn

class SeamlessApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "active": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Seamless"

    def apply(self, model, active):
        if not active:
            return (model, )
        
        model = model.clone()
        for layer in model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                layer.padding_mode = 'circular'
                print(layer)
        return (model, )
        

