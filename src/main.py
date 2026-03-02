import cv2
import numpy as np
import argparse
import sys

def process_video(video_path, max_frames=200):
    # This function processes the video to show optical flow using two methods:
    # 1. Lucas-Kanade (for tracking specific points)
    # 2. Farneback (for tracking the whole image densely)
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # If the video didn't open, let's try opening the webcam instead
    if not cap.isOpened():
        print("Couldn't open the video, falling back to your webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam didn't open either. Stopping.")
            sys.exit(1)

    # Read the very first frame of the video
    ret, old_frame = cap.read()
    if not ret:
        print("Couldn't read the first frame.")
        return

    # Resize the video so it runs faster on our computer
    width = int(old_frame.shape[1] * 0.5)
    height = int(old_frame.shape[0] * 0.5)
    old_frame = cv2.resize(old_frame, (width, height))

    # Convert the first frame to grayscale (black and white)
    # Optical flow methods usually just need the brightness/intensity of pixels, not colors
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # --- Setup for Lucas-Kanade (Point Tracking) ---
    # Find some good points to track in the first frame (corners are usually good)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Create an empty image just for drawing the colorful tracking lines
    mask_lk = np.zeros_like(old_frame)
    # Give us random colors for the 100 lines
    # Note: mask_lk fades each frame (see loop) so trails don't clutter the screen
    colors = np.random.randint(0, 255, (100, 3))
    
    # --- Setup for Farneback (Dense Tracking) ---
    # Create an image to hold the Farneback flow map in HSV color space
    # H = Hue (Direction of movement)
    # S = Saturation (Kept at max)
    # V = Value (How fast it's moving)
    hsv_fb = np.zeros_like(old_frame)
    hsv_fb[..., 1] = 255  # Set saturation to maximum
    
    frame_idx = 0
    print("Starting video processing. Press 'ESC' to exit.")
    
    # Loop over the video frames
    while frame_idx < max_frames:
        ret, current_frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
            
        current_frame = cv2.resize(current_frame, (width, height))
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # ---------------------------------------------------------
        # 1. Lucas-Kanade Tracking
        # ---------------------------------------------------------
        # Only try to track if we still have points left to track
        if p0 is not None and len(p0) > 0:
            # Calculate where the points moved to (p1)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, current_gray, p0, None, 
                winSize=(15, 15), maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Select only the points that were successfully tracked
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            else:
                # p1 being None means tracking completely failed — give empty numpy arrays
                good_new = np.empty((0, 2), dtype=np.float32)
                good_old = np.empty((0, 2), dtype=np.float32)
                
            # Draw the movement lines and points
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Draw the trail line
                mask_lk = cv2.line(mask_lk, (int(a), int(b)), (int(c), int(d)), colors[i % 100].tolist(), 2)
                # Draw the point itself on the current frame
                current_frame = cv2.circle(current_frame, (int(a), int(b)), 5, colors[i % 100].tolist(), -1)
                
            # Fade the trail mask slightly so old trails gradually disappear (avoids clutter)
            mask_lk = (mask_lk * 0.95).astype(np.uint8)
            
            # Combine the lines with the current frame
            img_lk = cv2.add(current_frame, mask_lk)
            
            # Save the new points to be the "old points" for the next frame
            # If good_new is empty (all points lost), reshape still works since it's a numpy array
            p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
        else:
            # Oh no, we lost all points! Let's find new ones to track.
            p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            # If we just reset, we don't draw any lines yet
            img_lk = current_frame.copy()

        # ---------------------------------------------------------
        # 2. Farneback Dense Optical Flow
        # ---------------------------------------------------------
        # Calculate the movement for every single pixel in the image
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, current_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Convert the horizontal (X) and vertical (Y) flow into speed and direction
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Color code it:
        # Hue (color) represents direction
        hsv_fb[..., 0] = ang * 180 / np.pi / 2
        # Value (brightness) represents speed
        hsv_fb[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert the HSV map back to normal colors (BGR) so we can display it
        img_fb = cv2.cvtColor(hsv_fb, cv2.COLOR_HSV2BGR)
        
        # ---------------------------------------------------------
        # 3. Create a Moving Objects Mask
        # ---------------------------------------------------------
        # Let's find which pixels moved more than 1.5 pixels distance
        _, binary_mask = cv2.threshold(mag, 1.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Clean up the mask using simple morphology (remove stray white dots, fill in small black holes)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask_clean = cv2.morphologyEx(binary_mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Convert mask to color image so we can put it side by side with the others
        binary_mask_colored = cv2.cvtColor(binary_mask_clean, cv2.COLOR_GRAY2BGR)

        # ---------------------------------------------------------
        # Display the Results side by side
        # ---------------------------------------------------------
        # Arrange the 4 images in a 2x2 grid
        top_row = np.hstack((img_lk, img_fb))
        original_color = cv2.cvtColor(old_gray, cv2.COLOR_GRAY2BGR) # show the plain grayscale
        bottom_row = np.hstack((original_color, binary_mask_colored))
        combined = np.vstack((top_row, bottom_row))
        
        # Add titles to each section
        cv2.putText(combined, 'Lucas-Kanade', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Farneback Dense Flow', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Original Grayscale', (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Binary Motion Mask', (width + 10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the combined window
        cv2.imshow('Optical Flow Analysis', combined)

        # Stop if the user pressed the 'ESC' key
        if cv2.waitKey(30) & 0xFF == 27:
            print("Escaping...")
            break

        # Get ready for the next frame
        old_gray = current_gray.copy()
        frame_idx += 1

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Optical Flow: Lucas-Kanade and Farneback')
    parser.add_argument('--video', type=str, default='0', help='Path to video file, or 0 for your webcam')
    parser.add_argument('--frames', type=int, default=300, help='How many frames to play before stopping')
    
    args = parser.parse_args()
    
    # If the user typed "0", we need it to be a number, not a string
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
        
    process_video(video_source, max_frames=args.frames)
