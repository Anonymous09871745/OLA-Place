import collections
import os
import os.path as osp

import numpy as np
import torch
import torch.optim as optim
import torch_geometric.transforms as T
import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from datapreparation.kitti360pose.utils import SCENE_NAMES_TRAIN, SCENE_NAMES_VAL
from models.branches import ObjectBranch
from training.losses import CCL_input_score


def create_save_dir():
    save_dir = "./checkpoints/object"
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_training_log(epoch, train_loss, val_acc, args, save_dir, is_best=False):
    log_file = osp.join(save_dir, "object_training_log.txt")
    with open(log_file, "a") as f:
        if epoch == 1:
            f.write(f"{'=' * 80}\n")
            f.write("Branch: OBJECT\n")
            f.write("Configuration:\n")
            f.write(f"  - epochs: {args.epochs}\n")
            f.write(f"  - learning_rate: {args.lr}\n")
            f.write(f"  - batch_size: {args.batch_size}\n")
            f.write(f"  - coarse_embed_dim: {args.coarse_embed_dim}\n")
            f.write(f"  - ranking_loss: {args.ranking_loss}\n")
            f.write(f"  - temperature: {args.temperature}\n")
            f.write(f"  - alpha: {args.alpha}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(
                f"{'Epoch':<8} {'TrainLoss':<12} "
                + "".join([f"{'Acc@' + str(k):<15}" for k in args.top_k])
                + f"{'Best@' + str(max(args.top_k)):<15} {'Status':<10}\n"
            )
            f.write(f"{'-' * 80}\n")

        best_marker = "★ BEST" if is_best else ""
        f.write(
            f"{epoch:<8} {train_loss:<12.6f} "
            + "".join([f"{val_acc[k]:<15.4f}" for k in args.top_k])
            + f"{val_acc[max(args.top_k)]:<15.4f} {best_marker:<10}\n"
        )


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train OLA-Place object branch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--no_pc_augment", action="store_true")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ranking_loss", type=str, default="CCL")
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--hungging_model", type=str, required=True)
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="exponential")
    parser.add_argument("--lr_step", type=float, default=10)
    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--use_riemannian", action="store_true")
    parser.add_argument("--riemannian_manifold", type=str, default="hyperbolic")
    return parser


def train_object_epoch(model, dataloader, criterion, optimizer, args, thresh=0.8):
    model.train()
    epoch_losses = []

    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        text_features = model.encode_text(batch["texts"])
        object_features, object_level_masks = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(text_features)

        aa_expanded = text_features.unsqueeze(1).expand(batch_size, batch_size, -1, -1)
        bb_transposed = object_features.transpose(1, 2)
        cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)
        aa_mask = torch.ones((batch_size, batch_size, text_features.size(1), 1), device=model.device)
        bb_mask = object_level_masks.transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_object = cc.max(dim=-1)[0].mean(dim=-1)

        mean_object_features = object_features.mean(dim=1)
        mean_object_features = mean_object_features / torch.norm(mean_object_features, p=2, dim=1, keepdim=True)
        mean_text_features = text_features.mean(dim=1)
        mean_text_features = mean_text_features / torch.norm(mean_text_features, p=2, dim=1, keepdim=True)
        dist_object = abs(torch.norm(mean_object_features.unsqueeze(1) - mean_text_features.unsqueeze(0), dim=2))
        dist_object = dist_object / dist_object.max()
        dist_object = pow((1 - dist_object + 1e-6), 1 / args.alpha)
        dist_object[dist_object > 1] = 1
        dist_object[dist_object < thresh] = thresh

        loss = criterion(scores_object, dist_object)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses)


@torch.no_grad()
def eval_object_branch(model, dataloader, args):
    model.eval()
    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    text_encodings = []
    cell_encodings = []
    cell_masks = []
    query_cell_ids = []
    db_cell_ids = []

    for batch in dataloader:
        text_features = model.encode_text(batch["texts"])
        text_encodings.append(text_features.cpu().numpy())
        query_cell_ids.extend(batch["cell_ids"])

    for batch in cells_dataloader:
        object_features, object_masks = model.encode_objects(batch["objects"], batch["object_points"])
        cell_encodings.append(object_features.cpu().numpy())
        cell_masks.append(object_masks.cpu().numpy())
        db_cell_ids.extend(batch["cell_ids"])

    text_encodings = np.vstack(text_encodings)
    cell_encodings = np.vstack(cell_encodings)
    cell_masks = np.vstack(cell_masks)
    query_cell_ids = np.array(query_cell_ids)
    db_cell_ids = np.array(db_cell_ids)

    scores = np.zeros((len(query_cell_ids), len(db_cell_ids)))
    for query_idx in range(len(query_cell_ids)):
        aa = torch.from_numpy(text_encodings[query_idx])
        bb = torch.from_numpy(cell_encodings)
        aa = aa.unsqueeze(0).expand(len(cell_encodings), args.num_mentioned, args.coarse_embed_dim)
        bb = bb.transpose(1, 2)
        cc = torch.matmul(aa, bb)
        aa_mask = torch.ones((args.num_mentioned, 1))
        bb_mask = torch.from_numpy(cell_masks).transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()

    accuracies = {k: [] for k in args.top_k}
    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])

    return accuracies


def main():
    parser = build_parser()
    args = EasyDict(vars(parser.parse_args()))
    if args.lr is None:
        args.lr = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.no_pc_augment:
        train_transform = T.FixedPoints(args.pointnet_numpoints)
        val_transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        train_transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.RandomRotate(120, axis=2),
            T.NormalizeScale(),
        ])
        val_transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale(),
        ])

    dataset_train = Kitti360CoarseDatasetMulti(
        args.base_path,
        SCENE_NAMES_TRAIN,
        train_transform,
        shuffle_hints=True,
        flip_poses=True,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=args.shuffle,
        num_workers=args.cpus,
    )

    dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    model = ObjectBranch(dataset_train.get_known_classes(), COLOR_NAMES_K360, args)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CCL_input_score(temperature=args.temperature, alpha=args.alpha)

    save_dir = create_save_dir()
    best_val_acc = -1
    best_model_path = None
    last_model_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_object_epoch(model, dataloader_train, criterion, optimizer, args)
        val_acc = eval_object_branch(model, dataloader_val, args)
        is_best = val_acc[max(args.top_k)] > best_val_acc
        if is_best:
            best_val_acc = val_acc[max(args.top_k)]

        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | " + " ".join([f"Acc@{k}={val_acc[k]:.4f}" for k in args.top_k]))
        save_training_log(epoch, train_loss, val_acc, args, save_dir, is_best)

        model_path = osp.join(save_dir, f"object_epoch{epoch}_acc{val_acc[max(args.top_k)]:.4f}.pth")
        if is_best:
            state_dict = model.state_dict()
            out = collections.OrderedDict()
            for item in state_dict:
                if "llm_model" not in item:
                    out[item] = state_dict[item]
            torch.save(out, model_path)
            best_model_path = model_path
            if last_model_path and osp.exists(last_model_path) and last_model_path != model_path:
                os.remove(last_model_path)
            last_model_path = model_path
            print(f"Saved best checkpoint to {model_path}")

    best_path = osp.join(save_dir, "object_best.pth")
    if best_model_path and best_model_path != best_path:
        torch.save(torch.load(best_model_path), best_path)
        print(f"Best model copied to {best_path}")


if __name__ == "__main__":
    main()
