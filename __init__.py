from .seamless import SeamlessApply, SeamlessVae, SeamlessKSampler, SeamlessKSamplerAdvanced

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

NODE_CLASS_MAPPINGS= {

    "SeamlessApply": SeamlessApply,
    "SeamlessKSampler": SeamlessKSampler,
    "SeamlessKSamplerAdvanced": SeamlessKSamplerAdvanced,
    "SeamlessVae": SeamlessVae

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamlessApply": "SeamlessApply",
    "SeamlessKSampler": "SeamlessKSampler",
    "SeamlessKSamplerAdvanced": "SeamlessKSamplerAdvanced",
    "SeamlessVae": "SeamlessVae"

}