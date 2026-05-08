import numpy as np
import torch
import torch_geometric.transforms as T
import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from datapreparation.kitti360pose.utils import KNOWN_CLASS, SCENE_NAMES_TEST, SCENE_NAMES_VAL
from evaluation.args import create_base_parser
from models.branches import ObjectBranch


def add_object_eval_args(parser):
    parser.add_argument("--object_checkpoint", type=str, required=True, help="Path to object-branch checkpoint")


@torch.no_grad()
def encode_object_branch(model, dataloader, cells_dataloader, args):
    n_queries = len(dataloader.dataset)
    n_cells = len(cells_dataloader.dataset)
    embed_dim = args.coarse_embed_dim

    text_enc_object = np.zeros((n_queries, args.num_mentioned, embed_dim), dtype=np.float32)
    cell_enc_object = np.zeros((n_cells, args.object_size, embed_dim), dtype=np.float32)
    cell_mask_object = np.zeros((n_cells, args.object_size, 1), dtype=np.float32)
    query_cell_ids = []
    db_cell_ids = []

    print("Encoding queries...")
    index_offset = 0
    for batch in tqdm.tqdm(dataloader, desc="Query encoding"):
        batch_size = len(batch["texts"])
        text_features = model.encode_text(batch["texts"])
        text_enc_object[index_offset:index_offset + batch_size] = text_features.cpu().numpy()
        query_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    print("Encoding cells...")
    index_offset = 0
    for batch in tqdm.tqdm(cells_dataloader, desc="Cell encoding"):
        batch_size = len(batch["cell_ids"])
        object_features, object_masks = model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_object[index_offset:index_offset + batch_size] = object_features.cpu().numpy()
        cell_mask_object[index_offset:index_offset + batch_size] = object_masks.cpu().numpy()
        db_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    return {
        "text_enc_object": text_enc_object,
        "cell_enc_object": cell_enc_object,
        "cell_mask_object": cell_mask_object,
        "query_cell_ids": np.array(query_cell_ids),
        "db_cell_ids": np.array(db_cell_ids),
    }


def compute_object_scores(encodings, args):
    scores_object = np.zeros((len(encodings["query_cell_ids"]), len(encodings["db_cell_ids"])))
    bb = torch.from_numpy(encodings["cell_enc_object"])
    bb_mask = torch.from_numpy(encodings["cell_mask_object"]).transpose(1, 2)
    aa_mask = torch.ones((args.num_mentioned, 1))

    for query_idx in tqdm.tqdm(range(len(encodings["query_cell_ids"])), desc="Computing scores"):
        aa = torch.from_numpy(encodings["text_enc_object"][query_idx])
        aa = aa.unsqueeze(0).expand(len(encodings["cell_enc_object"]), args.num_mentioned, args.coarse_embed_dim)
        cc = torch.matmul(aa, bb.transpose(1, 2))
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_object[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()

    return scores_object


def evaluate(scores, encodings, cells_dataset, query_poses_w, args):
    query_cell_ids = encodings["query_cell_ids"]
    db_cell_ids = encodings["db_cell_ids"]
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])

        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    return accuracies, accuracies_close


def main():
    parser = create_base_parser()
    add_object_eval_args(parser)
    args = EasyDict(vars(parser.parse_args()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    scene_names = SCENE_NAMES_TEST if args.use_test_set else SCENE_NAMES_VAL
    dataset = Kitti360CoarseDatasetMulti(args.base_path, scene_names, transform, shuffle_hints=False, flip_poses=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dataset = dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    model = ObjectBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model.load_state_dict(torch.load(args.object_checkpoint, map_location="cpu"), strict=False)
    model.to(device)
    model.eval()

    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataset.all_poses])
    encodings = encode_object_branch(model, dataloader, cells_dataloader, args)
    scores = compute_object_scores(encodings, args)
    accuracies, accuracies_close = evaluate(scores, encodings, cells_dataset, query_poses_w, args)

    print("\nObject-level coarse retrieval results")
    print("=" * 60)
    for k in args.top_k:
        print(f"Hit@{k:<2d}: {accuracies[k]:.4f} | Close@{k:<2d}: {accuracies_close[k]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
