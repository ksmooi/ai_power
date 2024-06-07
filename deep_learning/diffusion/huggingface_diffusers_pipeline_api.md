# A Comprehensive Guide to Stable Diffusion Pipeline

## Stable Diffusion Pipelines Introduction

Stable Diffusion is a text-to-image latent diffusion model created by researchers and engineers from CompVis, Stability AI, and LAION. Latent diffusion applies the diffusion process over a lower-dimensional latent space to reduce memory and compute complexity. This specific type of diffusion model was proposed in the paper *High-Resolution Image Synthesis with Latent Diffusion Models* by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

Stable Diffusion is trained on 512x512 images from a subset of the LAION-5B dataset. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and can run on consumer GPUs.

Task-Specific Applications of Stable Diffusion Pipelines:

| Pipeline                          | Supported tasks                                               |
|-----------------------------------|---------------------------------------------------------------|
| StableDiffusion                   | text-to-image                                                 |
| StableDiffusionImg2Img            | image-to-image                                                |
| StableDiffusionInpaint            | inpainting                                                    |
| StableDiffusionDepth2Img          | depth-to-image                                                |
| StableDiffusionUpscale            | super-resolution                                              |


## Overview of Pipeline Parameters
These parameters are crucial and frequently used in Stable Diffusion pipelines. It's essential to closely examine their definitions and the provided examples to fully understand how to effectively utilize them for optimal image generation results.

### Height and Width
The height and width parameters control the dimensions of the generated image in pixels. By default, the Stable Diffusion v1.5 model outputs 512x512 images, but you can change this to any size that is a multiple of 8.

### Negative Prompt
A negative prompt guides the model away from generating certain elements in an image. This is useful for quickly improving image quality and preventing the model from generating unwanted elements.

### Number of Inference Steps
`num_inference_steps` (int, optional, defaults to 50): This parameter controls the number of denoising steps. More denoising steps usually lead to a higher quality image but result in slower inference.

### Strength
The strength parameter measures how much noise is added to the base image, influencing how similar the output is to the base image.

- 📈 **High strength**: More noise is added, the denoising process takes longer, but the images are higher quality and more different from the base image.
- 📉 **Low strength**: Less noise is added, the denoising process is faster, but the image quality may not be as high and the generated image resembles the base image more.

### Guidance Scale
The guidance scale affects how closely the generated image aligns with the text prompt.

- 📈 **High guidance scale**: The prompt and generated image are closely aligned, resulting in a stricter interpretation of the prompt.
- 📉 **Low guidance scale**: The prompt and generated image are more loosely aligned, allowing for more varied output from the prompt.

### Padding Mask Crop
The `padding_mask_crop` parameter enhances inpainting image quality. When enabled, it crops the masked area with user-specified padding and crops the same area from the original image. Both the image and mask are then upscaled for inpainting and overlaid on the original image.

### Generator
A `torch.Generator` object ensures reproducibility by setting a manual seed. You can use a generator to create batches of images and iteratively improve on an image generated from a seed, as detailed in the "Improve image quality with deterministic generation" guide.

### Number of Images Per Prompt
`num_images_per_prompt` (int, optional, defaults to 1): This parameter specifies the number of images to generate per prompt.


## Examples of Using Stable Diffusion Pipelines

Here are five examples to guide you on how to effectively use the pipeline with properly configured parameters.
| Example                             | Description                                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------------------|
| StableDiffusionPipeline             | Generates images based on a text prompt, with parameters to control the image generation process. |
| StableDiffusionImg2ImgPipeline      | Generates images based on a text prompt and an initial image, with parameters to control the image generation process. |
| StableDiffusionInpaintPipeline      | Generates images based on a text prompt and an initial image with specific masked areas, with parameters to control the image generation process. |
| StableDiffusionDepth2ImgPipeline    | Generates images based on a text prompt, an initial image, and a depth map, with parameters to control the image generation process. |
| StableDiffusionUpscalePipeline      | Generates upscaled images based on a text prompt and an initial image, with parameters to control the image generation process. |

### StableDiffusionPipeline Example

This example demonstrates how to use the `StableDiffusionPipeline` to generate images based on a text prompt, with additional parameters to control the image generation process.

```python
from diffusers import StableDiffusionPipeline
import torch

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Define the parameters
prompt = "A serene landscape with mountains and a river"
negative_prompt = "city, buildings, pollution"
height = 512  # Height of the generated image in pixels
width = 512   # Width of the generated image in pixels
num_inference_steps = 100  # Number of denoising steps
guidance_scale = 7.5  # Higher value encourages model to follow the prompt closely
num_images_per_prompt = 2  # Number of images to generate per prompt
generator = torch.manual_seed(42)  # Set the random seed for reproducibility

# Generate images using the pipeline
images = pipeline(
    prompt=prompt,  # The text prompt guiding the image generation
    negative_prompt=negative_prompt,  # The text prompt guiding what to avoid in the image
    height=height,  # The height of the generated image
    width=width,  # The width of the generated image
    num_inference_steps=num_inference_steps,  # The number of denoising steps
    guidance_scale=guidance_scale,  # The guidance scale value
    num_images_per_prompt=num_images_per_prompt,  # The number of images to generate
    generator=generator  # The random seed generator
).images

# Display the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")
    img.show()
```

**Explanation of the Code**
1. **Import Necessary Modules**: We import `StableDiffusionPipeline` from `diffusers` and `torch` for tensor operations and setting random seeds.
2. **Load the Pipeline**: We load the `StableDiffusionPipeline` from a pre-trained model checkpoint. Here, "CompVis/stable-diffusion-v1-4" is used.
3. **Define Parameters**:
    - `prompt`: The text prompt that guides the image generation.
    - `negative_prompt`: The text prompt to specify what should not be included in the image.
    - `height` and `width`: Dimensions of the generated image.
    - `num_inference_steps`: The number of denoising steps, which impacts the quality and time taken for generation.
    - `guidance_scale`: Encourages the model to generate images closely linked to the text prompt.
    - `num_images_per_prompt`: The number of images to generate per prompt.
    - `generator`: A `torch.Generator` object to set the random seed for reproducibility.
4. **Generate Images**: We call the pipeline with the defined parameters. This generates images based on the specified prompt and parameters.
5. **Display and Save Images**: We iterate over the generated images, saving each one with a unique filename and displaying them.


### StableDiffusionImg2ImgPipeline Example

This example demonstrates how to use the `StableDiffusionImg2ImgPipeline` to generate images based on a text prompt and an initial image, with additional parameters to control the image generation process.

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# Load the stable diffusion image-to-image pipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Load the initial image to be used as a starting point
init_image = Image.open("path_to_your_image.jpg").convert("RGB")

# Define the parameters
prompt = "A futuristic cityscape with flying cars"
negative_prompt = "night, dark, gloomy"
strength = 0.75  # Indicates the extent to which the initial image should be transformed
num_inference_steps = 100  # Number of denoising steps
generator = torch.manual_seed(42)  # Set the random seed for reproducibility
num_images_per_prompt = 2  # Number of images to generate per prompt

# Generate images using the pipeline
images = pipeline(
    prompt=prompt,  # The text prompt guiding the image generation
    image=init_image,  # The initial image used as the starting point
    strength=strength,  # The extent to which the initial image is transformed
    num_inference_steps=num_inference_steps,  # The number of denoising steps
    guidance_scale=7.5,  # Higher value encourages model to follow the prompt closely
    negative_prompt=negative_prompt,  # The text prompt guiding what to avoid in the image
    num_images_per_prompt=num_images_per_prompt,  # The number of images to generate
    generator=generator  # The random seed generator
).images

# Display and save the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")
    img.show()
```

**Explanation of the Code**
1. **Import Necessary Modules**: Import the `StableDiffusionImg2ImgPipeline` from `diffusers`, `Image` from `PIL` for image handling, and `torch` for tensor operations and setting random seeds.
2. **Load the Pipeline**: Load the `StableDiffusionImg2ImgPipeline` from a pre-trained model checkpoint. Here, "CompVis/stable-diffusion-v1-4" is used.
3. **Load the Initial Image**: Load the initial image that will be used as the starting point for the image-to-image transformation.
4. **Define Parameters**:
    - `prompt`: The text prompt that guides the image generation.
    - `negative_prompt`: The text prompt to specify what should not be included in the image.
    - `strength`: Indicates the extent to which the initial image should be transformed (must be between 0 and 1).
    - `num_inference_steps`: The number of denoising steps, which impacts the quality and time taken for generation.
    - `generator`: A `torch.Generator` object to set the random seed for reproducibility.
    - `num_images_per_prompt`: The number of images to generate per prompt.
5. **Generate Images**: Call the pipeline with the defined parameters. This generates images based on the specified prompt and parameters.
6. **Display and Save Images**: Iterate over the generated images, save each one with a unique filename, and display them.


### StableDiffusionInpaintPipeline Example

This example demonstrates how to use the `StableDiffusionInpaintPipeline` to generate images based on a text prompt and an initial image with specific masked areas, with additional parameters to control the image generation process.

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# Load the stable diffusion inpainting pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Load the initial image and mask image
init_image = Image.open("path_to_your_image.jpg").convert("RGB")
mask_image = Image.open("path_to_your_mask_image.png").convert("L")  # Mask image should be single channel (luminance)

# Define the parameters
prompt = "A beautiful landscape with clear skies"
negative_prompt = "buildings, cars, people"
padding_mask_crop = 10  # The size of margin in the crop to be applied to the image and masking
strength = 0.8  # Indicates the extent to transform the reference image
num_inference_steps = 100  # Number of denoising steps
generator = torch.manual_seed(42)  # Set the random seed for reproducibility
num_images_per_prompt = 2  # Number of images to generate per prompt

# Generate images using the pipeline
images = pipeline(
    prompt=prompt,  # The text prompt guiding the image generation
    image=init_image,  # The initial image to be inpainted
    mask_image=mask_image,  # The mask image indicating areas to be repainted
    padding_mask_crop=padding_mask_crop,  # The size of margin in the crop to be applied to the image and masking
    strength=strength,  # The extent to transform the reference image
    num_inference_steps=num_inference_steps,  # The number of denoising steps
    guidance_scale=7.5,  # Higher value encourages model to follow the prompt closely
    negative_prompt=negative_prompt,  # The text prompt guiding what to avoid in the image
    num_images_per_prompt=num_images_per_prompt,  # The number of images to generate
    generator=generator  # The random seed generator
).images

# Display and save the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")
    img.show()
```

**Explanation of the Code**
1. **Import Necessary Modules**: Import the `StableDiffusionInpaintPipeline` from `diffusers`, `Image` from `PIL` for image handling, and `torch` for tensor operations and setting random seeds.
2. **Load the Pipeline**: Load the `StableDiffusionInpaintPipeline` from a pre-trained model checkpoint. Here, "CompVis/stable-diffusion-v1-4" is used.
3. **Load the Initial Image and Mask Image**: 
    - `init_image`: Load the initial image that will be used as the starting point for the inpainting process.
    - `mask_image`: Load the mask image indicating areas to be repainted. The mask image should be in a single channel (luminance) format.
4. **Define Parameters**:
    - `prompt`: The text prompt that guides the image generation.
    - `negative_prompt`: The text prompt to specify what should not be included in the image.
    - `padding_mask_crop`: The size of margin in the crop to be applied to the image and masking.
    - `strength`: Indicates the extent to which the initial image should be transformed (must be between 0 and 1).
    - `num_inference_steps`: The number of denoising steps, which impacts the quality and time taken for generation.
    - `generator`: A `torch.Generator` object to set the random seed for reproducibility.
    - `num_images_per_prompt`: The number of images to generate per prompt.
5. **Generate Images**: Call the pipeline with the defined parameters. This generates images based on the specified prompt, initial image, and mask image.
6. **Display and Save Images**: Iterate over the generated images, save each one with a unique filename, and display them.


### StableDiffusionDepth2ImgPipeline Example

This example demonstrates how to use the `StableDiffusionDepth2ImgPipeline` to generate images based on a text prompt, an initial image, and a depth map, with additional parameters to control the image generation process.

```python
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image
import torch

# Load the stable diffusion depth-to-image pipeline
pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Load the initial image and depth map
init_image = Image.open("path_to_your_image.jpg").convert("RGB")
depth_map = torch.load("path_to_your_depth_map.pt")  # Assuming depth map is stored as a torch tensor

# Define the parameters
prompt = "A futuristic cityscape with flying cars"
negative_prompt = "night, dark, gloomy"
strength = 0.75  # Indicates the extent to transform the reference image
num_inference_steps = 100  # Number of denoising steps
generator = torch.manual_seed(42)  # Set the random seed for reproducibility
num_images_per_prompt = 2  # Number of images to generate per prompt

# Generate images using the pipeline
images = pipeline(
    prompt=prompt,  # The text prompt guiding the image generation
    image=init_image,  # The initial image used as the starting point
    depth_map=depth_map,  # The depth map used as additional conditioning
    strength=strength,  # The extent to transform the reference image
    num_inference_steps=num_inference_steps,  # The number of denoising steps
    guidance_scale=7.5,  # Higher value encourages model to follow the prompt closely
    negative_prompt=negative_prompt,  # The text prompt guiding what to avoid in the image
    num_images_per_prompt=num_images_per_prompt,  # The number of images to generate
    generator=generator  # The random seed generator
).images

# Display and save the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")
    img.show()
```

**Explanation of the Code**
1. **Import Necessary Modules**: Import the `StableDiffusionDepth2ImgPipeline` from `diffusers`, `Image` from `PIL` for image handling, and `torch` for tensor operations and setting random seeds.
2. **Load the Pipeline**: Load the `StableDiffusionDepth2ImgPipeline` from a pre-trained model checkpoint. Here, "CompVis/stable-diffusion-v1-4" is used.
3. **Load the Initial Image and Depth Map**: 
    - `init_image`: Load the initial image that will be used as the starting point for the depth-to-image transformation.
    - `depth_map`: Load the depth map, which is a tensor containing depth information for the initial image.
4. **Define Parameters**:
    - `prompt`: The text prompt that guides the image generation.
    - `negative_prompt`: The text prompt to specify what should not be included in the image.
    - `strength`: Indicates the extent to which the initial image should be transformed (must be between 0 and 1).
    - `num_inference_steps`: The number of denoising steps, which impacts the quality and time taken for generation.
    - `generator`: A `torch.Generator` object to set the random seed for reproducibility.
    - `num_images_per_prompt`: The number of images to generate per prompt.
5. **Generate Images**: Call the pipeline with the defined parameters. This generates images based on the specified prompt, initial image, and depth map.
6. **Display and Save Images**: Iterate over the generated images, save each one with a unique filename, and display them.


### StableDiffusionUpscalePipeline Example

This example demonstrates how to use the `StableDiffusionUpscalePipeline` to generate upscaled images based on a text prompt and an initial image, with additional parameters to control the image generation process.

```python
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch

# Load the stable diffusion upscale pipeline
pipeline = StableDiffusionUpscalePipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Load the initial image to be upscaled
init_image = Image.open("path_to_your_image.jpg").convert("RGB")

# Define the parameters
prompt = "A highly detailed painting of a fantasy landscape"
negative_prompt = "low quality, blurry"
guidance_scale = 9.0  # A higher value encourages the model to generate images closely linked to the prompt
noise_level = 20  # The amount of noise to add to the image
num_inference_steps = 75  # The number of denoising steps
generator = torch.manual_seed(42)  # Set the random seed for reproducibility

# Generate images using the pipeline
images = pipeline(
    prompt=prompt,  # The text prompt guiding the image generation
    image=init_image,  # The initial image to be upscaled
    num_inference_steps=num_inference_steps,  # The number of denoising steps
    guidance_scale=guidance_scale,  # Higher value encourages model to follow the prompt closely
    noise_level=noise_level,  # The amount of noise to add to the image
    negative_prompt=negative_prompt,  # The text prompt guiding what to avoid in the image
    generator=generator  # The random seed generator
).images

# Display and save the generated images
for i, img in enumerate(images):
    img.save(f"upscaled_image_{i}.png")
    img.show()
```

**Explanation of the Code**
1. **Import Necessary Modules**: Import the `StableDiffusionUpscalePipeline` from `diffusers`, `Image` from `PIL` for image handling, and `torch` for tensor operations and setting random seeds.
2. **Load the Pipeline**: Load the `StableDiffusionUpscalePipeline` from a pre-trained model checkpoint. Here, "CompVis/stable-diffusion-v1-4" is used.
3. **Load the Initial Image**: Load the initial image that will be upscaled.
4. **Define Parameters**:
    - `prompt`: The text prompt that guides the image generation.
    - `negative_prompt`: The text prompt to specify what should not be included in the image.
    - `guidance_scale`: A higher value encourages the model to generate images closely linked to the prompt.
    - `noise_level`: The amount of noise to add to the image, influencing the upscaling effect.
    - `num_inference_steps`: The number of denoising steps, which impacts the quality and time taken for generation.
    - `generator`: A `torch.Generator` object to set the random seed for reproducibility.
5. **Generate Images**: Call the pipeline with the defined parameters. This generates images based on the specified prompt and initial image.
6. **Display and Save Images**: Iterate over the generated images, save each one with a unique filename, and display them.


## Conclusion
Stable Diffusion represents a significant advancement in text-to-image generation, leveraging latent diffusion models to produce high-quality images while maintaining computational efficiency. Developed by CompVis, Stability AI, and LAION, the model's application spans multiple tasks, including text-to-image, image-to-image, inpainting, depth-to-image, and super-resolution. Its ability to be trained on large datasets and run on consumer GPUs makes it accessible and practical for a wide range of users.

Understanding and utilizing the various parameters of Stable Diffusion pipelines is essential for achieving optimal results. Key parameters such as height and width, negative prompt, number of inference steps, strength, guidance scale, padding mask crop, generator, and the number of images per prompt play crucial roles in influencing the quality and characteristics of the generated images. By mastering these parameters, users can effectively harness the power of Stable Diffusion to create diverse and high-quality images tailored to their specific needs.

For more detailed information and practical examples, refer to the comprehensive documentation provided by Hugging Face. These resources offer valuable insights and guidance on maximizing the potential of Stable Diffusion pipelines.


## References
These references provide comprehensive documentation on the various Stable Diffusion pipelines available in Hugging Face's Diffusers library.
- [Stable Diffusion Pipeline Overview](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/overview)
- [Text-to-Image Pipeline](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/text2img)
- [Image-to-Image Pipeline](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/img2img)
- [Inpainting Pipeline](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/inpaint)
- [Depth-to-Image Pipeline](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/depth2img)
- [Upscale Pipeline](https://huggingface.co/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/upscale)
