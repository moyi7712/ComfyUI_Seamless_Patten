It change UNetModel and VAE Conv2d Layer into circular padding mode that make any text2image process generate seamless patten

Core code:
```python
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
```

<div class="image-container">
    <img src="./example/seamless.jpg" alt="Image semless">
    <img src="./example/seamless_tile.jpg" alt="Image seamless tile">
</div>

<!-- <!DOCTYPE html>
<html>
    <head>
        <style>
            .image-container {
                display: flex; 
                justify-content: center;
                align-items: center;
                flex-wrap: wrap; 
                margin: 0px; 
            }
            .image-container img {
                flex: 1; 
                width: calc(33.333% - 20px);
                border: none; 
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <div class="image-container">
            <img src="./example/seamless2.png" alt="Image 1">
            <img src="./example/seamless2.png" alt="Image 2">
        </div>
        <div class="image-container">
            <img src="./example/seamless2.png" alt="Image 3">
            <img src="./example/seamless2.png" alt="Image 4">
        </div>
    </body>
</html> -->
