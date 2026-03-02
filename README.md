# Assignment 1: Optical Flow and Video Dynamics
**Variant B: Educational Motion Analysis System (Tracking Module)**

## Overview
This repository contains a simple motion tracking module to compare two fundamental optical flow algorithms:
1. **Lucas-Kanade (LK)**: A sparse optical flow method that tracks keypoints.
2. **Farnebäck**: A dense optical flow method that estimates motion for every pixel.

The main script is located at `src/main.py`. It takes an input video (or webcam stream), computes the optical flow using both algorithms, and visualizes the results side-by-side.

## Setup and Usage

### Requirements
- Python 3.6+
- OpenCV (`cv2`)
- NumPy

### Running the script
To run the analysis on your webcam for 300 frames:
```bash
python src/main.py --video 0 --frames 300
```
To run on a specific video file:
```bash
python src/main.py --video path_to_your_video.mp4
```

## Implementation Details

### Parameters Used
**Lucas-Kanade Parameter Setup:**
- **Keypoint Detection (`goodFeaturesToTrack`)**:
  - `maxCorners`: 100
  - `qualityLevel`: 0.3
  - `minDistance`: 7
  - `blockSize`: 7
- **Optical Flow Tracking (`calcOpticalFlowPyrLK`)**:
  - `winSize`: (15, 15)
  - `maxLevel`: 2
  - `criteria`: `(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)`

**Farnebäck Parameter Setup:**
- **Dense Flow (`calcOpticalFlowFarneback`)**:
  - `pyr_scale`: 0.5 (image scale < 1 to build pyramids for each image)
  - `levels`: 3 (number of pyramid layers)
  - `winsize`: 15 (averaging window size)
  - `iterations`: 3 (iterations at each pyramid level)
  - `poly_n`: 5 (pixel neighborhood size to find polynomial expansion in each pixel)
  - `poly_sigma`: 1.2 (standard deviation of the Gaussian used to smooth derivatives)

### Filters and Motion Extraction
For Farnebäck, we extract the mask of moving objects using the **flow magnitude**.
1. Computed via `cv2.cartToPolar` to extract polar coordinates (magnitude and angle).
2. The mask is created using `cv2.threshold(mag, 1.5, 255, cv2.THRESH_BINARY)`.
3. An optional Morphological OPEN followed by CLOSE is applied with an `ELLIPSE (5, 5)` kernel to clean up noises in the binary mask so that moving objects are correctly captured without fragmentation.

## Comparative Study and Error Analysis

### 1. Sensitivity to Texture
- **Lucas-Kanade**: Highly dependent on the presence of corners and textured surfaces. If moving objects are plain (e.g., solid-color walls), LK feature detection might fail or track weakly.
- **Farnebäck**: Computes motion for the entire frame using polynomial expansion. Even with low texture, overlapping windows provide motion estimations, though they may lack precision compared to highly textured areas.

### 2. Sensitivity to Motion Blur and Global Changes
- **Lucas-Kanade**: Drops keypoints dynamically when they leave the frame or blur significantly across two successive frames rendering the template matching (LK assumption) invalid.
- **Farnebäck**: Generates large chaotic noise vectors in blurred areas. Sometimes motion blur is misinterpreted as a smooth motion over a large neighborhood.

### 3. Sensitivity to Shadows and Noise
- **Lucas-Kanade**: Features mapped on a moving shadow might get tracked causing incorrect object trajectory estimations.
- **Farnebäck**: Shadows are often tracked as part of the object because they provide a moving gradient. White noise in dark areas yields a high noisy magnitude map.

### 4. Mask Fragmentation
- In the Farnebäck method, the thresholded binary mask often exhibits fragmentation on objects that have uniform colors or move very slowly. Morphological operations (opening/closing) help to regroup these fragments, but very large texture-less objects can still leave holes in their center since only edges report strong magnitude.

## Conclusion: When to use which?
- **Use Lucas-Kanade (Sparse)** when you need fast, computationally inexpensive tracking of specific rigid points (e.g., traffic monitoring, face landmark tracking, feature-based SLAM). It's great if the objects have distinctive corners.
- **Use Farnebäck (Dense)** when you need to understand the motion of the entire scene, detect boundaries of moving non-rigid objects, or support background subtraction. It is much more computationally expensive but yields a comprehensive motion field.

## Visualizations
The script outputs a 2x2 grid containing:
1. **Lucas-Kanade**: Video stream overlaid with tracked keypoint trajectories (lines).
2. **Dense Flow Map**: HSV color-coded motion map (Hue = Direction, Value = Magnitude).
3. **Original View**: Grayscale view.
4. **Binary Motion Mask**: Thresholded and morphologically cleaned motion map showing active moving silhouettes.
