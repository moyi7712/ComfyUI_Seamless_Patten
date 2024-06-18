import torch.nn as nn



def vae_circular_hook_pre(module, input):
    if hasattr(module, 'padding_mode'):
        setattr(module, 'padding_mode', 'circular')


def vae_circular_hook(module, input, output):
    if hasattr(module, 'padding_mode'):
        setattr(module, 'padding_mode', 'zeros')
    
    if hasattr(module, 'circular_pre_hook'):
        getattr(module, 'circular_pre_hook').remove()
        delattr(module, 'circular_pre_hook')
    
    if hasattr(module, 'circular_hook'):
        getattr(module, 'circular_hook').remove()
        delattr(module, 'circular_hook')



def circular_hook_pre(module, input):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            setattr(layer, 'padding_mode', 'circular')

def circular_hook(module, input, output):
    time_emb = input[1]

    if time_emb>1:

        return
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            setattr(layer, 'padding_mode', 'zeros')

    if hasattr(module, 'circular_pre_hook'):
        getattr(module, 'circular_pre_hook').remove()
        delattr(module, 'circular_pre_hook')
    
    if hasattr(module, 'circular_hook'):
        getattr(module, 'circular_hook').remove()
        delattr(module, 'circular_hook')




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
        if not active:
            return (model, vae)

        model = model.clone()

        unet_prehook = model.model.diffusion_model.register_forward_pre_hook(circular_hook_pre)
        unet_hook = model.model.diffusion_model.register_forward_hook(circular_hook)
        setattr(model.model.diffusion_model, 'circular_pre_hook', unet_prehook)
        setattr(model.model.diffusion_model, 'circular_hook', unet_hook)
        
        patcher = vae.patcher.clone()
        for layer in patcher.model.modules():
            if (isinstance(layer, nn.Conv2d)):
                pre_hook = layer.register_forward_pre_hook(vae_circular_hook_pre)
                hook = layer.register_forward_hook(vae_circular_hook)
                setattr(layer, 'circular_pre_hook', pre_hook)
                setattr(layer, 'circular_hook', hook)
        vae.patcher = patcher
        vae.first_stage_model = patcher.model
        return (model, vae)
    
    @classmethod       
    def IS_CHANGED(s, **kwargs):
        return float("nan")

