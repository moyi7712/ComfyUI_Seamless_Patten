import torch.nn as nn
import functools

import comfy.samplers
# import comfy.sample
from nodes import common_ksampler



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


def circular_hook(model_sampling, module, input, output):
    time_emb = input[1]

    sigma = model_sampling.sigma(time_emb)

    if sigma.max()>model_sampling.percent_to_sigma(0.9):
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
        unet_hook = model.model.diffusion_model.register_forward_hook(functools.partial(circular_hook, model.model.model_sampling))
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
    


class SeamlessVae:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", ),
            }

        }
    
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "apply"
    CATEGORY = "Seamless"


    def apply(self, vae):        
        patcher = vae.patcher.clone()
        for layer in patcher.model.modules():
            if (isinstance(layer, nn.Conv2d)):
                pre_hook = layer.register_forward_pre_hook(vae_circular_hook_pre)
                hook = layer.register_forward_hook(vae_circular_hook)
                setattr(layer, 'circular_pre_hook', pre_hook)
                setattr(layer, 'circular_hook', hook)
        vae.patcher = patcher
        vae.first_stage_model = patcher.model

        return (vae, )
    
    @classmethod       
    def IS_CHANGED(s, **kwargs):
        return float("nan")
    

class SeamlessKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        padding_mode_list = []
        for layer in  model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                padding_mode_list.append(layer.padding_mode)
                layer.padding_mode = 'circular'

        ret =  common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        ind = 0
        for layer in  model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                padding_mode_list.append(layer.padding_mode)
                layer.padding_mode = padding_mode_list[ind]
                ind += 1

        return ret


class SeamlessKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        padding_mode_list = []
        for layer in  model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                padding_mode_list.append(layer.padding_mode)
                layer.padding_mode = 'circular'
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        ret = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

        ind = 0
        for layer in  model.model.diffusion_model.modules():
            if (isinstance(layer, nn.Conv2d)):
                padding_mode_list.append(layer.padding_mode)
                layer.padding_mode = padding_mode_list[ind]
                ind += 1
        return ret