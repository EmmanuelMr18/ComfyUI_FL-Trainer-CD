import torch

# This logic is created with AI, if you see a better solution
# please feel free to share it!
def transformImages(images):
  images_transformed = []

  # Process input_images to match the format of loaded images
  if isinstance(images, torch.Tensor):
    # If it's a single tensor, we need to split it into a list of tensors
    if images.dim() == 4:  # [batch, channels, height, width]
      images_transformed = [img.unsqueeze(0) for img in images]  # Split into list of [1, channels, height, width]
    elif images.dim() == 3:  # [channels, height, width]
      images_transformed = [images.unsqueeze(0)]  # Add batch dimension
    else:
      raise ValueError(f"Expected input tensor to have 3 or 4 dimensions, got {images.dim()}")
  elif isinstance(images, (list, tuple)):
    # If it's already a list or tuple, ensure each item is a tensor with correct shape
    images_transformed = []
    for img in images:
      if isinstance(img, torch.Tensor):
        if img.dim() == 3:
          images_transformed.append(img.unsqueeze(0))
        elif img.dim() == 4:
          images_transformed.append(img)
        else:
          raise ValueError(f"Expected tensor to have 3 or 4 dimensions, got {img.dim()}")
      else:
        images_transformed = images
        raise TypeError(f"Expected tensor, got {type(img)}")

  return images_transformed

  __all__ = ['transformImages']
