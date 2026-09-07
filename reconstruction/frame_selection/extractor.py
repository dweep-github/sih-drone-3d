import cv2
import os
import json
import numpy as np
from pathlib import Path

class FrameExtractor:
    def __init__(self, input_video, output_dir, target_frames=400, blur_threshold=100.0, sim_threshold=0.95):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.target_frames = target_frames
        self.blur_threshold = blur_threshold
        self.sim_threshold = sim_threshold
        
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def is_blurry(self, gray_frame):
        """Returns True if the frame variance is below the blur threshold."""
        variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        return variance < self.blur_threshold, variance

    def is_similar(self, hist1, hist2):
        """Compares two histograms using correlation. Returns True if similarity > threshold."""
        if hist1 is None or hist2 is None:
            return False
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity > self.sim_threshold

    def compute_histogram(self, gray_frame):
        """Computes a normalized histogram for the frame."""
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def process(self):
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.input_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps else 0

        # Calculate a baseline skip interval to aim for slightly more than the target 
        # (assuming we will drop some due to blur/similarity)
        base_interval = max(1, total_frames // (self.target_frames * 2))

        print(f"Video: {total_frames} frames @ {fps} FPS ({duration:.1f}s)")
        print(f"Sampling interval: {base_interval} (evaluating ~{total_frames // base_interval} frames)")

        saved_count = 0
        frame_idx = 0
        last_hist = None
        metadata_log = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only evaluate frames based on the interval
            if frame_idx % base_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Blur Check
                blurry, variance = self.is_blurry(gray)
                if not blurry:
                    # 2. Similarity Check
                    current_hist = self.compute_histogram(gray)
                    if not self.is_similar(last_hist, current_hist):
                        
                        # Save Frame
                        filename = f"{frame_idx:06d}.jpg"
                        filepath = self.frames_dir / filename
                        cv2.imwrite(str(filepath), frame)
                        
                        metadata_log.append({
                            "filename": filename,
                            "timestamp_sec": round(frame_idx / fps, 3),
                            "original_frame": frame_idx,
                            "laplacian_variance": round(variance, 2)
                        })
                        
                        last_hist = current_hist
                        saved_count += 1
                        print(f"Saved {filename} | Variance: {variance:.1f} | Total: {saved_count}")

                        # Stop if we hit our maximum ceiling to prevent overloading COLMAP
                        if saved_count >= self.target_frames * 1.5:
                            print("Reached upper limit of frame budget. Stopping extraction.")
                            break

            frame_idx += 1

        cap.release()

        # 3. Generate Metadata for Person 1
        metadata = {
            "source_video": os.path.basename(self.input_video),
            "original_fps": fps,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "selected_frames": saved_count,
            "blur_threshold_used": self.blur_threshold,
            "frames": metadata_log
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"\nExtraction complete. Saved {saved_count} clean frames to {self.frames_dir}")
        print("Metadata generated. Hand-off ready for Person 1.")

if __name__ == "__main__":
    # Test execution
    extractor = FrameExtractor(
        input_video="input/drone_video.mp4", 
        output_dir=".",
        target_frames=400
    )
    extractor.process()