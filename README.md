# ML Blendshapes

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training](#training)
   - [Training Process](#training-process)
4. [Using the Trained Model](#using-the-trained-model)
   - [Usage Process](#usage-process)
5. [Mathematical Explanation](#mathematical-explanation)
   - [Normalization](#normalization)
   - [Model Prediction](#model-prediction)
   - [Scaling & Masking](#scaling--masking)
6. [File Structure](#file-structure)
7. [License](#license)

## Overview
This package provides a machine learning pipeline for training models that predict blendshape displacements from neutral face positions. It includes scripts for training models on character-specific data and applying the trained models to generate displacement vectors.

## 🎥 Demo Video  
[➡️ Watch the ML Blendshape in Action](https://vimeo.com/1050592377)

## Installation
Ensure you have the required dependencies installed before using the scripts:
```sh
pip install numpy tensorflow
```

## Training
To train models for blendshape prediction, execute:
```sh
rez env ml_blendshapes -- train
```
This will process character-specific data and train a model for each target blendshape.

### Training Process
1. **Load Displacement Data:** Extracts blendshape vectors from JSON files.
2. **Normalize Input Data:** Scales data to a [-1, 1] range using min-max normalization.
3. **Define Neural Network:** A multi-layer perceptron with ReLU activations.
4. **Train the Model:** Uses mean squared error loss to optimize predictions.
5. **Save Model & Scaling Data:** Stores trained models and normalization parameters.

## Using the Trained Model
Once trained, you can generate blendshape displacements with:
```sh
python use_script.py
```
This script loads the trained models and applies them to input neutral positions.

### Usage Process
1. **Preprocess Input Data:** Loads a JSON file and normalizes values.
2. **Load Models & Predict Displacements:** Each target model computes a displacement vector.
3. **Apply Masking:** Ensures noise is suppressed by multiplying results with a mask.
4. **Save Results:** Outputs JSON with predicted displacements.

## Mathematical Explanation

The model predicts blendshape displacements based on normalized vertex positions. The process follows these steps:

### **1. Normalization**
To ensure consistent input values across different characters, we normalize the input vertex positions.

For each vertex $X$, we compute its normalized value using min-max scaling:

```math
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
```

where:
- $X_{min}$ and $X_{max}$ are the minimum and maximum displacement values for that vertex across the dataset.
- $X_{norm}$ is the scaled input, ranging between [-1, 1].

This transformation ensures that all input values are on the same scale, improving the stability of the neural network training.

### **2. Model Prediction**
The trained neural network maps normalized neutral positions $X_{norm}$ to blendshape displacements $Y_{pred}$:

```math
Y_{pred} = M(X_{norm})
```

where:
- $M$ represents the trained neural network.
- $Y_{pred}$ is the predicted displacement for a given blendshape.

Since the network was trained on normalized data, it naturally predicts outputs within a similar range.

### **3. Scaling the Displacement**
After obtaining the predicted blendshape displacement $Y_{pred}$, we apply a scaling factor to adjust its intensity:

```math
Y_{scaled} = Y_{pred} \times scale
```

This scaling was determined empirically through testing and ensures that the blendshapes have visually appropriate magnitudes.

### **4. Applying the Mask**
Each blendshape has a mask that determines which vertices should receive displacement updates. The mask is computed as:

```math
\text{Mask}[i] = \begin{cases} 
1, & \text{if } |X_{max}[i] - X_{min}[i]| > \epsilon \\
0, & \text{otherwise}
\end{cases}
```

where:
- $\epsilon$ is a small threshold ($10^{-5}$) to avoid numerical instability.
- The mask ensures that only vertices with significant displacement variations are modified.

Applying the mask to the scaled displacement gives the final output:

```math
Y_{final} = Y_{scaled} \times \text{Mask}
```

This ensures that small, noisy displacements are removed, improving the robustness of the results.


## File Structure
```
ml_blendshapes/
│-- blendshapes_data/    # vertex positions of each target for each characters
│-- normalization_data/  # Min/max values and masks
│-- model_training/      # Trained Keras models
│-- vector_data/         # Input displacement vectors
```

## License
```text
Custom License Agreement

Copyright (c) 2025 Felix Benicourt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
study, and modify the Software for personal, educational, or internal purposes,
subject to the following conditions:

1. Redistribution and Resale:
   - Redistribution, sublicensing, or resale of the Software, in whole or in part, 
     is strictly prohibited without prior written consent from the author.

2. Attribution:
   - This copyright notice and permission notice shall be included in all copies 
     or substantial portions of the Software.

3. No Warranty:
   - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
     FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL 
     THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER 
     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING 
     FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
     IN THE SOFTWARE.
```

