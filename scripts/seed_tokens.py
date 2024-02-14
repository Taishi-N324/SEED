# Adapted from https://huggingface.co/datasets/laion/laion_100m_vqgan_f8/blob/main/run_vqgan.py for vqgan_f16_16384

import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
import pyrootutils
import argparse
import traceback
import braceexpand
import warnings
import numpy as np
import pandas as pd
import webdataset as wds
import torch.multiprocessing as mp

from tqdm import tqdm
from timeit import default_timer as timer


warnings.filterwarnings("ignore", category=UserWarning)
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

ALLOWED_DATASETS = ["laion", "mmc4", "datacomp", "wiki", "grit", "obelics"]
    
EXPECTED_CHUNK_SIZE = 10000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms
from multiprocessing import Process, Queue, Manager
from queue import Empty


def transform_and_remove_keys_datacomp(sample):
    image, metadata, key, url = sample

    # CLIP transform without resizing
    image = transforms.functional.resize(image, (224, 224))
    image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    
    new_dictionary = {}
    new_dictionary['key'] = metadata['key']
    new_dictionary['caption'] = metadata['caption']
    # new_dictionary['uid'] = metadata['uid']
    new_dictionary['path'] = url
    return image, new_dictionary, key

def transform_and_remove_keys_wiki(sample):
    image, metadata, key, url = sample

    # CLIP transform without resizing
    image = transforms.functional.resize(image, (224, 224))
    image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    new_dictionary = {}
    new_dictionary['key'] = metadata['key']
    new_dictionary['path'] = url
    return image, new_dictionary, key

def remove_keys(sample):
    image, metadata, key = sample
    new_metadata = {}

    image = transforms.functional.resize(image, (224, 224))
    image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    keys_to_keep = ['caption', 'similarity']

    for k, v in metadata.items():
        if k in keys_to_keep:
            new_metadata[k] = v
    return image, new_metadata, key

def get_dataset(dataset_type, path, s3):
    if s3:
        path = f"pipe:aws s3 cp {path} -"

    if dataset_type == "laion":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg", "json")
        )
        dataset = dataset.map(remove_keys)

        return dataset
    elif dataset_type == "datacomp":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg;png;webp", "json", "__key__", "__url__")
        )
        dataset = dataset.map(transform_and_remove_keys_datacomp)

        return dataset
    elif dataset_type == "wiki":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg;png;webp", "json", "__key__", "__url__")
        )
        dataset = dataset.map(transform_and_remove_keys_wiki)

        return dataset
    elif dataset_type == "mmc4":

        def resize_image(sample):
            keys = ["png", "jpg", "jpeg"]
            for key in keys:
                if key in sample:
                    image = np.array(sample[key].resize((256, 256))).astype(np.float32)
                    image = image.transpose(2, 0, 1) / 255.0
                    sample["image"] = torch.from_numpy(image)
            return sample

        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .map(resize_image)
            .to_tuple("image", "__key__")
        )
        return dataset


def writer_worker(q, output_dir):
    
    while True:
        print("asdfafadfa")
        try:
            sample = q.get(timeout=100) 

            if sample is None:  # None is the signal to stop.
                break
            rows, embeddings = sample
            df = pd.DataFrame(rows)
            embeddings_cpu = embeddings.reshape(len(df), -1)
            df["seed_tokens"] = [item.tobytes() for item in embeddings_cpu]
            
            grouped = df.groupby("path")
            for path, group in grouped:
                basename = os.path.basename(path)
                output_path = os.path.join(
                    output_dir, os.path.splitext(basename)[0] + ".parquet"
                )
                # Remove the path column as it's no longer needed
                group = group.drop(columns=["path"])
                # df.drop(columns=["embeddings"], inplace=True)
                # df["seed_tokens"] = df["embeddings"].apply(lambda x: x.tobytes())
                
                # Check if parquet file exists and append or write accordingly
                if os.path.exists(output_path):
                    # Read existing parquet file into a dataframe to append new data
                    existing_df = pd.read_parquet(output_path)
                    updated_df = pd.concat([existing_df, group])
                    # Write/overwrite the parquet file with the updated dataframe
                    updated_df.to_parquet(output_path, index=False)
                else:
                    # Write the new dataframe to a parquet file directly
                    group.to_parquet(output_path, index=False)

        except Empty:
            continue

def process_chunk(
    rank,
    world_size,
    tokenizer_cfg_path,
    transform_cfg_path,
    paths,
    output_dir,
    num_workers,
    batch_size,
    s3,
    dataset_type,
    q,
):
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=rank, load_diffusion=False)

    # num_paths_per_chunk = int(np.ceil(len(paths) / world_size))
    # python seed_tokens.py -p "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{0000000..0000007}.tar" -o /p/fastdata/mmlaion/hummingbird/temp_seed -nw 32 -ng 4 -bs 2048
    # worker_paths = paths[
    #     rank * num_paths_per_chunk : min(len(paths), (rank + 1) * num_paths_per_chunk)
    # ]
    worker_paths = paths[rank::world_size]
    print (f"Rank: {rank} processing {len(worker_paths)} shards")

    dataset = get_dataset(dataset_type, worker_paths, s3)

    dataloader = torch.utils.data.DataLoader(
        dataset, #.batched(batch_size),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    rows = {}
    writer_p = Process(target=writer_worker, args=(q, output_dir))
    writer_p.start()

    num_chunks = len(worker_paths)


    for data, metas, key_ in tqdm(
                dataloader,
                total=int(np.ceil(EXPECTED_CHUNK_SIZE * num_chunks / batch_size)),
                desc=f"Rank : {rank}",
                position=rank,
                leave=False,
            ):
        image_tensor = data.to(rank)
        image_ids = tokenizer.encode_image(image_torch=image_tensor).cpu().numpy()

        for k, v in metas.items():
            if type(v) == torch.Tensor:
                v = v.cpu().numpy().tolist()
            rows[k] = v
        rows["__key__"] = key_
        q.put((rows, image_ids))
        rows = {}

    q.put(None)  # signal for the writer worker to stop
    writer_p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        help="/path/to/images/{0000..1111}.tar",
    )
    parser.add_argument(
        "-s3",
        action="store_true",
        help="Pass this flag if using s3 bucket",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for *.parquet files with the code column",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="Number of workers per gpu for the dataloader",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per gpu for the dataloader",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="datacomp",
        help="Type of dataset used. Can be 'laion' or 'mmc4'",
    )
    args = parser.parse_args()


    pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

    tokenizer_cfg_path = '../configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    transform_cfg_path = '../configs/transform/clip_transform.yaml'

    paths = list(braceexpand.braceexpand(args.paths))

    start = timer()

    if args.dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Dataset must be one of {ALLOWED_DATASETS}, got {args.dataset}"
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    manager = Manager()
    q = manager.Queue()

    if args.num_gpus > 1:
        mp.spawn(
            process_chunk,
            args=(
                args.num_gpus,
                tokenizer_cfg_path,
                transform_cfg_path,
                paths,
                args.output_dir,
                args.num_workers,
                args.batch_size,
                args.s3,
                args.dataset,
                q,
            ),
            nprocs=args.num_gpus,
        )
    else:
        process_chunk(0, 1, tokenizer_cfg_path, transform_cfg_path, paths, args.output_dir, args.num_workers, args.batch_size, args.s3, args.dataset, q)

    print(f"Processing {len(paths)} shards took {timer() - start} seconds")


if __name__ == "__main__":
    main()


# python seed_tokens.py -p "/p/scratch/ccstdl/nakamura2/en_wiki_img2dataset/{00000..00013}.tar" -o /p/fastdata/mmlaion/hummingbird/temp_wiki_seed -nw 4 -ng 4 -bs 2048 --dataset wiki

# python seed_tokens.py -p "/p/fastdata/mmlaion/obelics_img/obelics-train-00000-of-01335/{00000..00020}.tar" -o /p/fastdata/mmlaion/hummingbird/temp_obelics_seed -nw 4 -ng 4 -bs 2048 --dataset wiki