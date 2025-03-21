import argparse
import torch
import torch.nn as nn  # 🔥 Added missing import
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta import create_model
from utils.utils import create_logger

def inference(model, config, input_joints, padding_mask, out_len=12):
    model.eval()
    
    with torch.no_grad():
        pred_joints = model(input_joints, padding_mask)

    output_joints = pred_joints[:, -out_len:]

    return output_joints


def evaluate_ade_fde(model, modality_selection, dataloader, bs, config, logger, return_all=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 9
    assert out_F == 12
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    ade_batch = 0 
    fde_batch = 0
    batch_id = 0

    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])

        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)

        out_joints = out_joints.cpu()
        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), out_F, 1, 2)    

        for k in range(len(out_joints)):
            person_out_joints = out_joints[k, :, 0:1]
            person_pred_joints = pred_joints[k, :, 0:1]

            gt_xy = person_out_joints[:, 0, :2]
            pred_xy = person_pred_joints[:, 0, :2]

            sum_ade = 0
            for t in range(out_F):
                dist_ade = np.linalg.norm(gt_xy[t].detach().cpu().numpy() - pred_xy[t].detach().cpu().numpy())
                sum_ade += dist_ade
            
            sum_ade /= out_F
            ade_batch += sum_ade

            dist_fde = np.linalg.norm(gt_xy[-1].detach().cpu().numpy() - pred_xy[-1].detach().cpu().numpy())
            fde_batch += dist_fde

        batch_id += 1

    ade = ade_batch / ((batch_id - 1) * bs + len(out_joints))
    fde = fde_batch / ((batch_id - 1) * bs + len(out_joints))

    return ade, fde


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. One of [train, test, valid]")
    parser.add_argument("--modality", type=str, default="traj+all", help="Available modality combinations")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ################################
    # Load checkpoint
    ################################

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    config = ckpt['config']

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################

    model = create_model(config, logger)

    # Fix state_dict mismatches by ignoring incompatible keys
    checkpoint_state_dict = ckpt['model']
    model_state_dict = model.state_dict()

    # Remove mismatched layers before loading
    for key in [
        "double_id_encoder.learned_encoding.weight",
        "bb3d_encoder.learned_encoding.weight",
        "bb2d_encoder.learned_encoding.weight",
        "pose3d_encoder.learned_encoding.weight",
        "pose2d_encoder.learned_encoding.weight"
    ]:
        if key in checkpoint_state_dict and key in model_state_dict:
            logger.warning(f"Skipping {key} due to shape mismatch: {checkpoint_state_dict[key].shape} vs {model_state_dict[key].shape}")
            del checkpoint_state_dict[key]

    # Load all compatible layers
    model.load_state_dict(checkpoint_state_dict, strict=False)

    # ✅ Reinitialize removed embedding layers
    with torch.no_grad():
        model.double_id_encoder.learned_encoding = nn.Embedding(10, 64).to(config["DEVICE"])
        model.bb3d_encoder.learned_encoding = nn.Embedding(10, 128).to(config["DEVICE"])
        model.bb2d_encoder.learned_encoding = nn.Embedding(10, 128).to(config["DEVICE"])
        model.pose3d_encoder.learned_encoding = nn.Embedding(10, 128).to(config["DEVICE"])
        model.pose2d_encoder.learned_encoding = nn.Embedding(10, 128).to(config["DEVICE"])

    logger.info("Model loaded successfully.")

    ################################
    # Load data
    ################################

    # Set 9-frame input + 12-frame prediction
    config['TRAIN']['input_track_size'] = 9
    config['TRAIN']['output_track_size'] = 12
    in_F, out_F = 9, 12

    name = config['DATA']['train_datasets']
    
    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F + out_F), track_cutoff=in_F)

    bs = config['TRAIN']['batch_size']
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    ade, fde = evaluate_ade_fde(model, args.modality, dataloader, bs, config, logger, return_all=True)

    print('ADE: ', ade)
    print('FDE: ', fde)
