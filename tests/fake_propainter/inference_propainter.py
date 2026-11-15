"""Stand-in for ProPainter's inference_propainter.py: the same arguments, input reading and output layout, but the
masked pixels are filled with cv2.inpaint. Read from sczhou/ProPainter main; anything the real script would refuse,
this refuses too. Test-only extras, all through environment variables:
FAKE_PROPAINTER_LOG appends one JSON line per run; FAKE_PROPAINTER_FILL=mean fills with the run's average colour;
FAKE_PROPAINTER_BREAK=exit fails like a crash, =short saves one frame too few."""

import argparse
import json
import os
import sys

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees')
parser.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask')
parser.add_argument('-o', '--output', type=str, default='results')
parser.add_argument("--resize_ratio", type=float, default=1.0)
parser.add_argument('--height', type=int, default=-1)
parser.add_argument('--width', type=int, default=-1)
parser.add_argument('--mask_dilation', type=int, default=4)
parser.add_argument("--ref_stride", type=int, default=10)
parser.add_argument("--neighbor_length", type=int, default=10)
parser.add_argument("--subvideo_length", type=int, default=80)
parser.add_argument("--raft_iter", type=int, default=20)
parser.add_argument('--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'])
parser.add_argument('--scale_h', type=float, default=1.0)
parser.add_argument('--scale_w', type=float, default=1.2)
parser.add_argument('--save_fps', type=int, default=24)
parser.add_argument('--save_frames', action='store_true')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

if args.video.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
    sys.exit("the stand-in reads image folders only")
if args.mode != 'video_inpainting':
    sys.exit("the stand-in does inpainting only")
if os.environ.get("FAKE_PROPAINTER_BREAK") == "exit":
    sys.exit("torch.OutOfMemoryError: CUDA out of memory.")

# frames: every file in the folder, in sorted name order, read with cv2 (a non-image file crashes the real one too)
video_name = os.path.basename(args.video)
frames = [cv2.imread(os.path.join(args.video, f)) for f in sorted(os.listdir(args.video))]
frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
if len(frames) < 2:  # the real one's RAFT reshapes l_t - 1 = 0 flows and raises
    raise RuntimeError("cannot reshape tensor of 0 elements")
out_size = frames[0].shape[1], frames[0].shape[0]
size = out_size
if args.width != -1 and args.height != -1:
    size = (args.width, args.height)
if args.resize_ratio != 1.0:
    size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
frames = [f if f.shape[1::-1] == size else cv2.resize(f, size) for f in frames]  # PIL bicubic in the real one

# masks: one image for every frame, or a folder with one per frame; any non-zero value means fill
if args.mask.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
    masks = [cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)] * len(frames)
else:
    masks = [cv2.imread(os.path.join(args.mask, f), cv2.IMREAD_GRAYSCALE) for f in sorted(os.listdir(args.mask))]
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # scipy binary_dilation's default structure
masks = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
if args.mask_dilation > 0:
    masks = [cv2.dilate(m.astype(np.uint8), cross, iterations=args.mask_dilation) > 0 for m in masks]

if os.environ.get("FAKE_PROPAINTER_FILL") == "mean":
    mean = np.mean([f[~m].mean(axis=0) for f, m in zip(frames, masks)], axis=0)
    filled = [np.broadcast_to(np.rint(mean).astype(np.uint8), f.shape) for f in frames]
else:
    filled = [cv2.inpaint(f, m.astype(np.uint8), 5, cv2.INPAINT_TELEA) for f, m in zip(frames, masks)]
comp = [np.where(m[..., None], p, f) for f, m, p in zip(frames, masks, filled)]

if args.save_frames:
    root = os.path.join(args.output, video_name, 'frames')
    os.makedirs(root, exist_ok=True)
    keep = len(comp) - (os.environ.get("FAKE_PROPAINTER_BREAK") == "short")
    for idx, f in enumerate(comp[:keep]):
        f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(root, str(idx).zfill(4) + '.png'), cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
# shortcut: the real one also writes masked_in.mp4 and inpaint_out.mp4 here; nothing reads them, so they are skipped

if os.environ.get("FAKE_PROPAINTER_LOG"):
    with open(os.environ["FAKE_PROPAINTER_LOG"], "a") as log:
        log.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "frames": len(frames), "size": list(out_size),
                              "corners": [f[0, 0].tolist() for f in frames]}) + "\n")
print(f'\nAll results are saved in {os.path.join(args.output, video_name)}')
