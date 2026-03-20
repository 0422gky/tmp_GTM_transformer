import argparse
import os
import sys
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset


def _torch_load_trusted(path: str):
    """
    PyTorch 2.6+ 默认 torch.load(weights_only=True) 会拒绝加载包含 numpy/非纯权重对象的 .pt。
    本脚本加载的是数据字典/ckpt（你一般是信任来源的），因此显式 weights_only=False。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容老版本 PyTorch（没有 weights_only 参数）
        return torch.load(path, map_location="cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = _torch_load_trusted(checkpoint_path)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)


def _extract_hparams(ckpt: Dict) -> Dict:
    if not isinstance(ckpt, dict):
        return {}
    if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
        return ckpt["hyper_parameters"]
    if "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        return ckpt["hparams"]
    return {}


def _read_split_df(data_folder: str, split: str) -> pd.DataFrame:
    csv_path = os.path.join(data_folder, f"{split}.csv")
    # parse_dates 用于满足 ZeroShotDataset.preprocess_data() 里的时间运算
    df = pd.read_csv(csv_path, parse_dates=["release_date"])
    return df


def _prepare_metadata_df(df: pd.DataFrame, split_value: str) -> pd.DataFrame:
    meta_cols = [
        "external_code",
        "season",
        "category",
        "release_date",
        "day",
        "week",
        "month",
        "year",
        "image_path",
        "color",
        "fabric",
        "extra",
    ]
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metadata columns missing in csv: {missing}")

    meta_df = df[meta_cols].copy()
    # 要求 release_date 作为字符串保留
    if np.issubdtype(meta_df["release_date"].dtype, np.datetime64):
        meta_df["release_date"] = meta_df["release_date"].dt.strftime("%Y-%m-%d")
    meta_df.insert(0, "split", split_value)
    meta_df = meta_df.reset_index(drop=True)
    return meta_df


def _ensure_item_embedding(
    fused_feature: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    fused_feature 运行时实际 shape 可能是：
    - (B, 1, H)
    - (B, T, H)
    - (T, B, H)  (则需要 transpose)
    """
    shape = tuple(fused_feature.shape)
    if fused_feature.dim() == 2:
        # (B, H)
        if fused_feature.shape[0] != batch_size:
            raise ValueError(f"Unexpected fused_feature shape: {shape}, batch_size={batch_size}")
        item_embedding = fused_feature
        return item_embedding, shape

    if fused_feature.dim() != 3:
        raise ValueError(f"Unexpected fused_feature dim={fused_feature.dim()}, shape={shape}")

    B = batch_size
    # 尝试把 fused_feature 统一成 (B, T, H)
    if fused_feature.shape[0] == B:
        # already (B, T, H) or (B, 1, H)
        bf = fused_feature
    elif fused_feature.shape[1] == B:
        # (T, B, H) -> (B, T, H)
        bf = fused_feature.transpose(0, 1)
    else:
        raise ValueError(f"Cannot infer batch dim from fused_feature shape={shape} with batch_size={B}")

    # bf: (B, T, H)
    T = bf.shape[1]
    if T == 1:
        item_embedding = bf[:, 0, :]
    else:
        item_embedding = bf.mean(dim=1)

    return item_embedding, shape


@torch.no_grad()
def export_for_df(
    split_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    model: GTM,
) -> None:
    meta_df = _prepare_metadata_df(df, split_name)

    # 用于模型的 df：preprocess_data 会 inplace drop 列，因此用副本
    df_for_model = df.copy(deep=True)

    dataset = ZeroShotDataset(
        data_df=df_for_model,
        img_root=os.path.join(args.data_folder, "images"),
        gtrends=args.gtrends,
        cat_dict=args.cat_dict,
        col_dict=args.col_dict,
        fab_dict=args.fab_dict,
        trend_len=args.trend_len,
    )

    # 复用原项目 preprocessing（它会一次性把 images/gtrends/gtrend 缓存在 TensorDataset 里）
    tensor_dataset = dataset.preprocess_data()

    loader = DataLoader(
        tensor_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 禁止 shuffle，确保 tensor 顺序与 meta_df 行顺序严格对齐
        num_workers=args.num_workers,
    )

    n = len(df_for_model)
    emb_matrix = None  # (N, H)
    sales_matrix = np.zeros((n, 12), dtype=np.float32)

    offset = 0
    model.eval()
    first_print_done = False

    for batch in loader:
        # TensorDataset item order:
        # item_sales, categories, colors, fabrics, temporal_features, gtrends, images
        item_sales, category, color, fabric, temporal_features, gtrends, images = batch

        # Move to device
        item_sales = item_sales.to(args.device)
        category = category.to(args.device)
        color = color.to(args.device)
        fabric = fabric.to(args.device)
        temporal_features = temporal_features.to(args.device)
        gtrends = gtrends.to(args.device)
        images = images.to(args.device)

        pred, fused_feature = model(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            return_embedding=True,
        )
        # fused_feature: decoder cross-attn output
        B = item_sales.shape[0]
        item_embedding, fused_shape = _ensure_item_embedding(fused_feature, batch_size=B)

        # runtime check / logging
        if not first_print_done:
            print(f"[{split_name}] fused_feature.shape(runtime)={fused_shape}, item_embedding.shape={tuple(item_embedding.shape)}")
            first_print_done = True

        item_emb_np = item_embedding.detach().cpu().numpy().astype(np.float32)
        sales_np = item_sales.detach().cpu().numpy().astype(np.float32)

        if emb_matrix is None:
            H = item_emb_np.shape[1]
            emb_matrix = np.zeros((n, H), dtype=np.float32)

        bs = item_emb_np.shape[0]
        emb_matrix[offset : offset + bs] = item_emb_np
        sales_matrix[offset : offset + bs] = sales_np
        offset += bs

    if emb_matrix is None:
        raise RuntimeError("No batches processed; cannot export embeddings.")
    if offset != n:
        raise RuntimeError(f"Export alignment error: processed_rows={offset}, expected_rows={n}")

    # Build output dataframe
    sales_cols = [f"sales_wk_{i}" for i in range(12)]
    sales_df = pd.DataFrame(sales_matrix, columns=sales_cols)

    H = emb_matrix.shape[1]
    emb_cols = [f"emb_{i:03d}" for i in range(H)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)

    final_df = pd.concat([meta_df, sales_df, emb_df], axis=1)

    out_csv_name = f"{split_name}_item_embeddings.csv"
    out_npy_name = f"{split_name}_item_embeddings.npy"
    out_meta_csv_name = f"{split_name}_item_embedding_meta.csv"

    os.makedirs(args.output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(args.output_dir, out_csv_name), index=False)
    np.save(os.path.join(args.output_dir, out_npy_name), emb_matrix)
    pd.concat([meta_df, sales_df], axis=1).to_csv(os.path.join(args.output_dir, out_meta_csv_name), index=False)


@torch.no_grad()
def export_for_split(
    split_name: str,
    args: argparse.Namespace,
    model: GTM,
) -> None:
    df = _read_split_df(args.data_folder, split_name)
    export_for_df(split_name=split_name, df=df, args=args, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path/to/model.ckpt or .pth")
    parser.add_argument("--data_folder", type=str, default="dataset/", help="dataset root folder")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (NOTE: this project masks are hard-coded to cuda)")

    # 当 checkpoint 不包含 hyper_parameters 时，这些用于兜底
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)

    args = parser.parse_args()

    # device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but this project generates masks on cuda. Please run with GPU or adjust model code.")
    device_str = f"cuda:{args.gpu_num}" if args.device.startswith("cuda") else "cpu"
    args.device = torch.device(device_str)

    # load csv embeddings metadata (and dicts/models inputs)
    cat_dict = _torch_load_trusted(os.path.join(args.data_folder, "category_labels.pt"))
    col_dict = _torch_load_trusted(os.path.join(args.data_folder, "color_labels.pt"))
    fab_dict = _torch_load_trusted(os.path.join(args.data_folder, "fabric_labels.pt"))

    args.cat_dict = cat_dict
    args.col_dict = col_dict
    args.fab_dict = fab_dict

    # Load gtrends
    args.gtrends = pd.read_csv(
        os.path.join(args.data_folder, "gtrends.csv"),
        index_col=[0],
        parse_dates=True,
    )

    # Load checkpoint + hyper parameters (if available)
    ckpt = _torch_load_trusted(args.checkpoint)
    hparams = _extract_hparams(ckpt)

    def get_h(k, default):
        return hparams.get(k, default)

    model = GTM(
        embedding_dim=get_h("embedding_dim", args.embedding_dim),
        hidden_dim=get_h("hidden_dim", args.hidden_dim),
        output_dim=get_h("output_dim", args.output_dim),
        num_heads=get_h("num_heads", args.num_attn_heads),
        num_layers=get_h("num_layers", args.num_hidden_layers),
        use_text=get_h("use_text", args.use_text),
        use_img=get_h("use_img", args.use_img),
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=get_h("trend_len", args.trend_len),
        num_trends=get_h("num_trends", args.num_trends),
        gpu_num=args.gpu_num,
        use_encoder_mask=get_h("use_encoder_mask", args.use_encoder_mask),
        autoregressive=get_h("autoregressive", args.autoregressive),
    )
    model.to(args.device)
    _load_checkpoint(model, args.checkpoint)

    # Ensure trend_len is available for dataset
    args.trend_len = get_h("trend_len", args.trend_len)

    splits_to_run: List[str]
    if args.split == "all":
        splits_to_run = ["train", "test", "all"]
    else:
        splits_to_run = [args.split]

    for sp in splits_to_run:
        if sp == "all":
            # 合并 train + test 并导出；外部要求 split 字段为 all
            train_df = _read_split_df(args.data_folder, "train")
            test_df = _read_split_df(args.data_folder, "test")
            df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            export_for_df(split_name="all", df=df_all, args=args, model=model)
        else:
            export_for_split(sp, args, model)

    print("Export finished.")


if __name__ == "__main__":
    main()