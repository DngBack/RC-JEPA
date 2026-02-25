# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Run I-JEPA encoder inference: load pretrained weights and extract features for image(s).

import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image

import yaml

from src.helper import init_model
import src.models.vision_transformer as vit


def get_transform(crop_size=224):
    return T.Compose([
        T.Resize(crop_size + 32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_encoder_from_checkpoint(checkpoint_path, device, config):
    """Build encoder from config and load encoder weights from checkpoint."""
    meta = config.get('meta', {})
    data = config.get('data', {})
    mask = config.get('mask', {})

    model_name = meta.get('model_name', 'vit_base')
    patch_size = mask.get('patch_size', 16)
    crop_size = data.get('crop_size', 224)
    pred_depth = meta.get('pred_depth', 6)
    pred_emb_dim = meta.get('pred_emb_dim', 384)

    encoder, predictor = init_model(
        device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'encoder' not in checkpoint:
        raise KeyError(f'Checkpoint must contain "encoder" key; keys: {list(checkpoint.keys())}')
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    encoder.eval()
    return encoder, crop_size


def main():
    parser = argparse.ArgumentParser(description='I-JEPA encoder inference: extract features from image(s).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth.tar checkpoint (e.g. IN1K-vit.h.14-300e.pth.tar)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML that matches the checkpoint (e.g. configs/in1k_vith14_ep300.yaml)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to a single image or a directory of images')
    parser.add_argument('--output', type=str, default='features.pt',
                        help='Output path for features .pt file (default: features.pt)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Max batch size (default: 0 = process all images in one batch; use 1 for low VRAM)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: cuda if available else cpu)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    encoder, crop_size = load_encoder_from_checkpoint(args.checkpoint, device, config)
    transform = get_transform(crop_size)

    # Collect image paths
    if os.path.isfile(args.image):
        image_paths = [args.image]
    elif os.path.isdir(args.image):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [
            os.path.join(args.image, n)
            for n in sorted(os.listdir(args.image))
            if os.path.splitext(n)[1].lower() in exts
        ]
        if not image_paths:
            raise SystemExit(f'No images found in directory: {args.image}')
    else:
        raise SystemExit(f'Not a file or directory: {args.image}')

    # Load and preprocess
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        tensors.append(transform(img))
    all_tensors = torch.stack(tensors)

    batch_size = args.batch_size if args.batch_size > 0 else len(image_paths)
    feature_chunks = []
    for i in range(0, len(image_paths), batch_size):
        batch = all_tensors[i : i + batch_size].to(device)
        with torch.no_grad():
            feats = encoder(batch, masks=None)
        feature_chunks.append(feats.cpu())
    features = torch.cat(feature_chunks, dim=0)

    # features: (N, num_patches, embed_dim)
    torch.save({'features': features, 'paths': image_paths}, args.output)
    print(f'Saved features shape {features.shape} to {args.output}')


if __name__ == '__main__':
    main()
