# **Homework 1: Optical Flow and Video Dynamics**

### **Objective**

Master the practical application of optical flow for motion analysis, learn to build complete video pipelines including flow computation, filtering, warping, and stabilization. The student chooses **one** of two engineering tasks: building a mini-system for **camera stabilization** or creating a **motion tracking module** analyzing the trajectories and stability of Lucas-Kanade and Farnebäck methods.

---

# **Variant A: Camera Stabilization Mini-System**

### **Task**

1. Choose a video with noticeable camera shake (phone recording, action camera, GoPro, handheld panorama).
2. Compute dense optical flow (Farnebäck) or sparse optical flow (LK) and estimate global camera motion across frames (affine model or homography).
3. Smooth the motion trajectory with a low-pass filter (moving window, exponential smoothing, or Kalman filter optionally).
4. Perform **motion compensation** by warping each frame to the stabilized trajectory.
5. Generate the stabilized video.
6. Build "before/after" visual comparisons and conduct **error analysis**: which areas stabilize poorly and why (optical flow, shadows, motion blur, lack of texture, global model errors).

---

# **Variant B: Educational Motion Analysis System (Tracking Module)**

### **Task**

1. Choose any video with several moving objects.
2. Implement computation of sparse optical flow (Lucas-Kanade) and dense optical flow (Farnebäck).
3. For LK:
   - find keypoints;
   - track trajectories over 50–200 frames;
   - visualize motion in the frame coordinate system.
4. For Farnebäck:
   - build the motion field;
   - extract moving objects through the flow magnitude;
   - obtain binary motion masks.
5. Compare the sensitivity of the methods to texture, motion blur, shadows, and noise.
6. Conduct **qualitative and quantitative error analysis**: where LK loses points, where dense flow outputs noise, where the motion mask fragments.
7. Prepare a brief study: what features would make a student choose LK or Farnebäck in a real task.

---

# **Submission Requirements**

GitHub Repository:

* `src/` or a notebook with the code for computing optical flow and the entire pipeline.
* `README.md` describing the chosen variant (A or B), parameters, applied filters, and main observations.
* Visualizations:
  - for Variant A: exact "before/after" frames, camera motion trajectory plots, warping examples;
  - for Variant B: LK trajectories, dense flow maps, motion masks.
* For Variant A — the final stabilized video.
* For Variant B — a comparative report on the differences between LK and Farnebäck.

---

# **Grading Criteria**

| Points | Criterion                                                                      |
| ------ | ------------------------------------------------------------------------------ |
| 0–3    | Completeness: all stages of the chosen variant (A or B) are implemented.       |
| 0–3    | Code: clean, reproducible, correctly uses OpenCV.                              |
| 0–2    | Analysis: qualitative error analysis, results interpretation, conclusions.     |
| 0–2    | Repository: neatness, readability, visualization examples, clear README.       |

**Maximum: 10 points.**

---
