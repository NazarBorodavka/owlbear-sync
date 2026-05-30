import subprocess
import os
import tempfile

class RUNETagDetector:
    def __init__(self, detector_path="./runetag_detector"):
        # Try to find the detector relative to the current working directory
        full_path = os.path.abspath(detector_path)
        
        # If not found, try to find it relative to this script's directory
        if not os.path.exists(full_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, os.path.basename(detector_path))
            
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Detector executable not found. Searched at: {os.path.abspath(detector_path)} and {full_path}")
            
        self.detector_path = full_path

    def detect(self, image_path, model_paths, intrinsics=None, output_img=None, 
               minarea=100, maxarea=10000, minroundness=0.3, maxmse=0.3):
        """
        Runs the RUNEtag detector.
        :param image_path: Path to the input image.
        :param model_paths: Single path or list of paths to .txt model files (or codes.txt).
        :param intrinsics: Optional dict with camera intrinsics {'fx', 'fy', 'cx', 'cy'}
        :param output_img: Optional path to save the debug image.
        :param minarea: Minimum ellipse area.
        :param maxarea: Maximum ellipse area.
        :param minroundness: Minimum ellipse roundness.
        :param maxmse: Maxmimum ellipse contour error.
        :return: List of detections, each a dict with 'idx', 'angle', 'R', 'T'
        """
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # Use temporary files for output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.pose', delete=False) as tmp_pose:
            pose_file_path = tmp_pose.name
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.points', delete=False) as tmp_points:
            points_file_path = tmp_points.name

        cmd = [
            self.detector_path, 
            "--img", os.path.abspath(image_path),
            "--minarea", str(minarea),
            "--maxarea", str(maxarea),
            "--minroundness", str(minroundness),
            "--maxmse", str(maxmse)
        ]
        
        for m in model_paths:
            cmd.extend(["-m", os.path.abspath(m)])
        
        cmd.extend(["--posefile", pose_file_path])
        cmd.extend(["--pointsfile", points_file_path])

        if output_img:
            cmd.extend(["--tagsimg", os.path.abspath(output_img)])

        if intrinsics:
            if 'fx' in intrinsics: cmd.extend(["--fx", str(intrinsics['fx'])])
            if 'fy' in intrinsics: cmd.extend(["--fy", str(intrinsics['fy'])])
            if 'cx' in intrinsics: cmd.extend(["--cx", str(intrinsics['cx'])])
            if 'cy' in intrinsics: cmd.extend(["--cy", str(intrinsics['cy'])])
            # Check if tuning params were passed in the dict instead (for backward compatibility)
            if 'minarea' in intrinsics: cmd.extend(["--minarea", str(intrinsics['minarea'])])
            if 'maxarea' in intrinsics: cmd.extend(["--maxarea", str(intrinsics['maxarea'])])
            if 'minroundness' in intrinsics: cmd.extend(["--minroundness", str(intrinsics['minroundness'])])
            if 'maxmse' in intrinsics: cmd.extend(["--maxmse", str(intrinsics['maxmse'])])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            detections = self._parse_output_files(pose_file_path, points_file_path)
            return detections
        except subprocess.CalledProcessError as e:
            # Only print the first line to avoid spamming the console with NTL exceptions
            err_msg = e.stderr.strip().split('\n')[0] if e.stderr else str(e)
            print(f"RuneTag Detector Error: {err_msg}")
            return []
        finally:
            if os.path.exists(pose_file_path):
                os.remove(pose_file_path)
            if os.path.exists(points_file_path):
                os.remove(points_file_path)

    def _parse_output_files(self, pose_path, points_path):
        detections = {}
        
        # Parse Pose File
        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 14: continue
                    idx = int(parts[0])
                    detections[idx] = {
                        'idx': idx,
                        'angle_deg': float(parts[1]),
                        'R': [
                            [float(parts[2]), float(parts[3]), float(parts[4])],
                            [float(parts[5]), float(parts[6]), float(parts[7])],
                            [float(parts[8]), float(parts[9]), float(parts[10])]
                        ],
                        'T': [float(parts[11]), float(parts[12]), float(parts[13])]
                    }
        
        # Parse Points File (for 2D centers)
        if os.path.exists(points_path):
            with open(points_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    idx = int(parts[0])
                    if idx in detections:
                        detections[idx]['x'] = float(parts[1])
                        detections[idx]['y'] = float(parts[2])
                    else:
                        # If for some reason it's in points but not pose
                        detections[idx] = {
                            'idx': idx,
                            'x': float(parts[1]),
                            'y': float(parts[2])
                        }
                        
        return list(detections.values())

if __name__ == "__main__":
    # Example usage
    import sys
    
    detector = RUNETagDetector()
    
    # Assuming we are running from the project root
    image = "20260514_001656.jpg"
    models = ["test_tags/tag_26.txt"]
    
    print(f"Detecting tags in {image}...")
    results = detector.detect(image, models, minarea=25, output_img="debug_result.jpg")
    
    if not results:
        print("No tags found.")
    for r in results:
        print(f"Found Tag {r['idx']}!")
        print(f"  Rotation Matrix:\n{r['R']}")
        print(f"  Translation Vector: {r['T']}")
        print("-" * 20)
