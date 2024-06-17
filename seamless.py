import torch.nn as nn

class SeamlessApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "vae": ("VAE", ),
                "active": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "apply"
    CATEGORY = "Seamless"

    def apply(self, model, vae, active):
        model = model.clone()
        for layer in model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                if active:
                    layer.padding_mode = 'circular'
                else:
                    layer.padding_mode = 'zeros'

        
        patcher = vae.patcher.clone()
        for layer in patcher.model.modules():
            if (isinstance(layer, nn.Conv2d)):
                if active:
                    layer.padding_mode = 'circular'
                else:
                    layer.padding_mode = 'zeros'
        vae.patcher = patcher
        vae.first_stage_model = patcher.model
        return (model, vae)
        

