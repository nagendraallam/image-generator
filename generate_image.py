from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None)
pipe = pipe.to("cuda")

prompt = "A fantasy landscape with mountains and a river"
image = pipe(prompt).images[0]

image.save("landscape.png")
