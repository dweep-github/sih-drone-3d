import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class DynamicObjectMasker:
    def __init__(self, frames_dir="frames", output_dir="."):
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(output_dir) / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO Classes we want to ignore: 
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 1, 2, 3, 5, 7] 
        
        print("Loading YOLO11 Nano Segmentation model...")
        # Automatically downloads YOLO11 segmentation weights on first run
        self.model = YOLO("yolo11n-seg.pt")
        
    def process(self):
        # Sort frames so ByteTrack accurately tracks movement over time
        frame_files = sorted([str(f) for f in self.frames_dir.glob("*.jpg")])
        
        if not frame_files:
            print(f"No frames found in {self.frames_dir}. Run extractor.py first.")
            return
            
        print(f"Tracking and generating masks for {len(frame_files)} frames...")
        
        results = self.model.track(
            source=frame_files,
            tracker="bytetrack.yaml",
            classes=self.target_classes,
            persist=True,
            stream=True 
        )
        
        tracking_log = []

        for result, frame_path_str in zip(results, frame_files):
            frame_path = Path(frame_path_str)
            height, width = result.orig_shape
            
            # COLMAP Rule: White (255) = Keep pixel, Black (0) = Ignore pixel
            colmap_mask = np.full((height, width), 255, dtype=np.uint8)
            frame_detections = []

            if result.masks is not None and result.boxes is not None:
                for i in range(len(result.boxes)):
                    mask_tensor = result.masks.data[i]
                    box = result.boxes[i]
                    mask_np = mask_tensor.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Paint the dynamic object black on our mask
                    colmap_mask[mask_resized > 0.5] = 0
                    
                    track_id = int(box.id[0]) if box.id is not None else -1
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    frame_detections.append({
                        "object_id": track_id,
                        "class": self.model.names[class_id],
                        "confidence": round(conf, 2)
                    })

            # Save mask with the exact same name as the frame (but .png)
            mask_filename = frame_path.with_suffix('.png').name
            cv2.imwrite(str(self.masks_dir / mask_filename), colmap_mask)
            
            tracking_log.append({
                "frame": frame_path.name,
                "detections": frame_detections
            })
            
            print(f"Masked {frame_path.name} | Dynamic Objects Found: {len(frame_detections)}")

        with open(self.masks_dir.parent / "tracking.json", "w") as f:
            json.dump({"frames": tracking_log}, f, indent=4)
            
        print(f"\nSuccess! Mask generation complete. Output saved to {self.masks_dir}")

if __name__ == "__main__":
    masker = DynamicObjectMasker()
    masker.process()