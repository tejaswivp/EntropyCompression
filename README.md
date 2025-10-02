# EntropyCompression
This repository explores entropy-based image analysis and simulation of human eye fixations using Levy walks. The project applies information-theoretic and stochastic processes to study how humans perceive and explore visual scenes.

## ✨ Features
- Compute entropy maps from images using local neighborhoods  
- Visualize entropy-based segmentation and masking  
- Experiment with entropy thresholds for edge detection  
- Simulate fixational eye movements with:  
  - Levy random walks  
  - Entropy maximization heuristics  
  - Time-minimization constraints  
- Generate 3D manifolds of fixation trajectories with entropy values  
- Analyze image collections for global entropy distribution and center bias  

## 📂 Project Structure
- `entropy_analysis`: Functions to compute entropy maps, thresholds, and masks  
- `fixation_simulation`: Functions for Levy-walk-based fixation modeling  
- `visualization`: Plotting routines for entropy maps, thresholds, and 3D manifolds  
- `Images_all/`: Example images used for experiments  

## 🛠 Requirements
Install dependencies with:
```bash
pip install numpy matplotlib scikit-image pillow scipy
```

## 🚀 Usage
### 1. Entropy Maps
```python
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt

image = rgb2gray(imread("Images_all/Chagall_IandTheVillage.jpg"))
entropy_image = entropy(image, disk(6))

plt.imshow(entropy_image, cmap="magma")
plt.title("Entropy Map")
plt.show()
```

### 2. Entropy Thresholding
```python
from entropy_analysis import threshold_checker
threshold_checker(image)
```

### 3. Simulated Eye Fixations with Levy Walks
```python
from fixation_simulation import simulate_fixations
from PIL import Image
import numpy as np

image = np.array(Image.open("Images_all/Sunday-1888-90-by-Paul-Signac.jpg").convert("L"))
fixations = simulate_fixations(image, num_fixations=200, scale=10, neighborhood_size=10)
```

### 4. 3D Manifold of Fixations & Entropy
```python
from mpl_toolkits.mplot3d import Axes3D

# fixations = simulate_fixations(...)
# entropies = compute_entropy_along_path(fixations)

ax.plot(fixations[:,0], fixations[:,1], entropies, marker='o')
```

### 5. Batch Entropy Analysis
```python
from entropy_analysis import analyze_images_in_folder
entropies = analyze_images_in_folder("Images_all/")
```

## 📊 Example Outputs
- Entropy maps highlighting texture-rich regions  
- Thresholded masks isolating entropy-dense areas  
- Fixation simulations overlayed on input images  
- 3D entropy manifolds showing fixation trajectory vs information content  
- Entropy distribution histograms across datasets  

## 📚 Theoretical Background
- **Entropy**: Shannon entropy measures unpredictability in pixel intensities, highlighting visually informative regions  
- **Levy Walks**: Models human saccades and fixational movements as long-tailed random jumps  
- **Visual Attention**: Combines entropy-driven saliency with stochastic dynamics to approximate human eye gaze  

## 🔮 Future Work
- Integrate CNN-based saliency prediction  
- Combine entropy-driven fixation models with reinforcement learning  
- Apply to real eye-tracking datasets for validation  
