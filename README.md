# Neural Style Transfer with VGG19

This project implements Neural Style Transfer using a pre-trained **VGG19** model in TensorFlow. The objective is to generate a new image that preserves the **content** of one image while adopting the **style** of another.

---

## Features

- Uses **VGG19** pretrained weights for feature extraction.
- Computes **content loss** and **style loss** using intermediate layers.
- Utilizes **Gram Matrix** for style representation.
- Trains the generated image using **gradient descent** and **Adam optimizer**.
- GPU-compatible using TensorFlow.
- Intermediate results are visualized during training.

---

## Workflow

1. **Load and preprocess images**
   - Resize to 400×400 and normalize pixel values.
2. **Load the VGG19 model**
   - Use `include_top=False` and freeze all layers.
3. **Extract features**
   - Use specific convolutional layers to extract content and style features.
4. **Loss computation**
   - Content loss: difference between high-level feature maps.
   - Style loss: based on Gram matrix comparison across selected layers.
5. **Optimization**
   - Train the image with backpropagation using Adam optimizer.
6. **Display results**
   - Show the original content, style, and the generated image.

---

## Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.x  
- NumPy  
- Pillow  
- Matplotlib  

---

## Model Details

- **Style layers:**
  - `block1_conv1`
  - `block2_conv1`
  - `block3_conv1`
  - `block4_conv1`
  - `block5_conv1`

- **Content layer:**
  - `block5_conv4`

- **Loss weights:**
  - `alpha = 10` (content weight)
  - `beta = 40` (style weight)

- **Optimizer:** `Adam`

- **Training steps:** 4001
