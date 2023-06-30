import argparse
import json
import os
import shutil
from collections import defaultdict
from inspect import signature
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set

import torch

from huggingface_hub import CommitInfo, CommitOperationAdd, Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.pipelines.base import infer_framework_load_model


class AlreadyExists(Exception):
    pass


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(model_id: str, folder: str, revision: str = "main") -> List["CommitOperationAdd"]:
    filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin.index.json", revision=revision)
    with open(filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []
    for filename in filenames:
        pt_filename = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)

        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(folder, sf_filename)
        convert_file(pt_filename, sf_filename)
        local_filenames.append(sf_filename)

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)

    operations = [
        CommitOperationAdd(path_in_repo=local.split("/")[-1], path_or_fileobj=local) for local in local_filenames
    ]

    return operations


def convert_single(model_id: str, folder: str) -> List["CommitOperationAdd"]:
    pt_filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")

    sf_name = "model.safetensors"
    sf_filename = os.path.join(folder, sf_name)
    convert_file(pt_filename, sf_filename)
    operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
    return operations


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def create_diff(pt_infos: Dict[str, List[str]], sf_infos: Dict[str, List[str]]) -> str:
    errors = []
    for key in ["missing_keys", "mismatched_keys", "unexpected_keys"]:
        pt_set = set(pt_infos[key])
        sf_set = set(sf_infos[key])

        pt_only = pt_set - sf_set
        sf_only = sf_set - pt_set

        if pt_only:
            errors.append(f"{key} : PT warnings contain {pt_only} which are not present in SF warnings")
        if sf_only:
            errors.append(f"{key} : SF warnings contain {sf_only} which are not present in PT warnings")
    return "\n".join(errors)

def previous_pr(api: "HfApi", model_id: str, pr_title: str) -> Optional["Discussion"]:
    try:
        discussions = api.get_repo_discussions(repo_id=model_id)
    except Exception:
        return None
    for discussion in discussions:
        if discussion.status == "open" and discussion.is_pull_request and discussion.title == pr_title:
            return discussion


def convert_generic(model_id: str, folder: str, filenames: Set[str], revision: str = "main") -> List["CommitOperationAdd"]:
    operations = []

    extensions = set([".bin", ".ckpt"])
    for filename in filenames:
        prefix, ext = os.path.splitext(filename)
        if ext in extensions:
            pt_filename = hf_hub_download(model_id, filename=filename, revision=revision)
            dirname, raw_filename = os.path.split(filename)
            if raw_filename == "pytorch_model.bin":
                # XXX: This is a special case to handle `transformers` and the
                # `transformers` part of the model which is actually loaded by `transformers`.
                sf_in_repo = os.path.join(dirname, "model.safetensors")
            else:
                sf_in_repo = f"{prefix}.safetensors"
            sf_filename = os.path.join(folder, sf_in_repo)
            convert_file(pt_filename, sf_filename)
            operations.append(CommitOperationAdd(path_in_repo=sf_in_repo, path_or_fileobj=sf_filename))
    return operations


def convert(api: "HfApi", model_id: str, force: bool = False, revision: str = "main") -> Optional["CommitInfo"]:
    pr_title = "Adding `safetensors` variant of this model"
    info = api.model_info(model_id, revision=revision)

    def is_valid_filename(filename):
        return len(filename.split("/")) > 1 or filename in ["pytorch_model.bin", "diffusion_pytorch_model.bin"]
    filenames = set(s.rfilename for s in info.siblings if is_valid_filename(s.rfilename))

    print(filenames)
    with TemporaryDirectory() as d:
        folder = os.path.join(d, repo_folder_name(repo_id=model_id, repo_type="models"))
        os.makedirs(folder)
        new_pr = None
        try:
            operations = None
            pr = previous_pr(api, model_id, pr_title)

            library_name = getattr(info, "library_name", None)
            if any(filename.endswith(".safetensors") for filename in filenames) and not force:
                raise AlreadyExists(f"Model {model_id} is already converted, skipping..")
            elif pr is not None and not force:
                url = f"https://huggingface.co/{model_id}/discussions/{pr.num}"
                new_pr = pr
                raise AlreadyExists(f"Model {model_id} already has an open PR check out {url}")
            else:
                print("Convert generic")
                operations = convert_generic(model_id, folder, filenames, revision=revision)

            if operations:
                new_pr = api.create_commit(
                    repo_id=model_id,
                    operations=operations,
                    commit_message=pr_title,
                    create_pr=True,
                )
                print(f"Pr created at {new_pr.pr_url}")
            else:
                print("No files to convert")
        finally:
            shutil.rmtree(folder)
        return new_pr


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create the PR even if it already exists of if the model was already converted.",
    )
    parser.add_argument(
    	"revision",
    	default="main",
    	help="Branch to convert. E.g. main, fp16, bf16"
    )
    args = parser.parse_args()
    model_id = args.model_id
    revision = args.revision
    api = HfApi()
    convert(api, model_id, force=args.force, revision=revision)
