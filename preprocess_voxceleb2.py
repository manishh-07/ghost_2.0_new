
import os
import cv2
import h5py
import torch
import numpy as np
import face_alignment
from insightface.app import FaceAnalysis
from repos.stylematte.stylematte.models import StyleMatte
from src.utils.crops import wide_crop_face, crop_face, emoca_crop
import onnxruntime as ort
from torchvision import transforms

# Unset MPLBACKEND to avoid Matplotlib backend conflict
os.environ.pop('MPLBACKEND', None)

def preprocess_video(video_path, output_h5_path, face_app, fa, segment_model, parsing_session, chunk_size=100, compress=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    frames, keypoints, masks, emoca_features, idx_68, parsings = [], [], [], [], [], []

    # Parsing function (performed on CPU)
    mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
    std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]
    input_name = parsing_session.get_inputs()[0].name
    output_names = [output.name for output in parsing_session.get_outputs()]
    
    def infer_parsing(img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # CPU tensor
        normalized_img = ((img_tensor[:, [2, 1, 0], ...] / 255. - torch.from_numpy(mean)) / torch.from_numpy(std)).numpy().astype(np.float32)
        result = parsing_session.run(output_names, {input_name: normalized_img})[0]
        return torch.from_numpy(result)  # Return as CPU tensor

    # StyleMatte normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure BGR format
        frame = frame[:, :, ::-1]

        # Face detection
        dets = face_app.get(frame)
        if len(dets) != 1:
            print(f"Frame {frame_count}: {'No face' if len(dets) == 0 else 'Multiple faces'} detected, skipping")
            continue

        kps = dets[0]['kps']

        # Crop faces
        face_wide = wide_crop_face(frame, kps, crop_size=512)
        face_arc = crop_face(frame, kps, crop_size=112)

        # Get 68 keypoints
        kpts = fa.get_landmarks_from_image(face_wide)
        if kpts is None or len(kpts) != 1 or kpts[0].shape[0] != 68:
            print(f"Frame {frame_count}: Invalid keypoints, skipping")
            continue
        kpts = kpts[0]

        # EMOCA features
        face_emoca = emoca_crop(face_wide[np.newaxis, ...], kpts[np.newaxis, ...])
        face_emoca = (face_emoca[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

        # Segmentation mask
        face_wide_torch = torch.from_numpy(face_wide).permute(2, 0, 1).cuda() / 255.0
        input_t = normalize(face_wide_torch).unsqueeze(0).float()
        with torch.no_grad():
            mask = segment_model(input_t)[0][0].cpu().numpy()

        # Parsing
        parsing = infer_parsing(face_wide).numpy()  # Convert to NumPy after CPU processing

        # Store data
        frames.append(frame)
        keypoints.append(kpts)
        masks.append(mask)
        emoca_features.append(face_emoca)
        idx_68.append(frame_count)
        parsings.append(parsing)

        frame_count += 1
        if frame_count % chunk_size == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()

    # Save to .h5 file
    if frames:
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('frames', data=np.array(frames), compression="gzip" if compress else None)
            f.create_dataset('keypoints', data=np.array(keypoints))
            f.create_dataset('masks', data=np.array(masks))
            f.create_dataset('emoca', data=np.array(emoca_features))
            f.create_dataset('idx_68', data=np.array(idx_68))
            f.create_dataset('parsing', data=np.array(parsings))
        print(f"Saved {output_h5_path} with {len(frames)} frames")
    else:
        print(f"No valid frames processed for {video_path}")

def main():
    # Initialize models
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda:0')

    segment_model = StyleMatte()
    segment_model.load_state_dict(
        torch.load('/content/ghost-2.0/repos/stylematte/stylematte/checkpoints/stylematte_synth.pth', map_location='cpu')
    )
    segment_model.cuda()
    segment_model.eval()

    parsing_session = ort.InferenceSession('/content/ghost-2.0/weights/segformer_B5_ce.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # # Root directory for all videos
    # video_root = '/content/ghost-2.0/src/data/test/mp4'
    # output_root = '/content/ghost-2.0/src/data/test/mp4'

    # Root directory for all videos in the train directory
    video_root = '/content/ghost-2.0/src/data/train/mp4'
    output_root = '/content/ghost-2.0/src/data/train/mp4'

    # Recursively process all .mp4 files
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                # Create output path preserving the directory structure
                relative_path = os.path.relpath(video_path, video_root)
                h5_path = os.path.join(output_root, relative_path.replace('.mp4', '.h5'))
                os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                if os.path.exists(video_path):
                    print(f"Processing {video_path}...")
                    preprocess_video(video_path, h5_path, face_app, fa, segment_model, parsing_session)
                else:
                    print(f"Video file {video_path} not found")

if __name__ == "__main__":
    main()
