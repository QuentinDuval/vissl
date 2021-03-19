# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
from typing import Iterable, NamedTuple, Sequence


try:
    from git import Repo
except ImportError:
    raise ValueError("This script requires gitpython: pip install gitpython")


class FileDiff(NamedTuple):
    change_type: str
    file_name: str


def get_index_diff_file_names(repo_dir: str) -> Iterable[FileDiff]:
    """
    Return the list of file diff in the staging area (the index)
    """
    repo = Repo(repo_dir)
    for file_diff in repo.head.commit.diff():
        yield FileDiff(change_type=file_diff.change_type, file_name=file_diff.a_path)


def get_working_diff_file_names(repo_dir: str) -> Iterable[FileDiff]:
    """
    Return the list of file diffs in the working tree (not added)
    """
    repo = Repo(repo_dir)
    for file_diff in repo.index.diff(None):
        yield FileDiff(change_type=file_diff.change_type, file_name=file_diff.a_path)
    for path in repo.untracked_files:
        yield FileDiff(change_type="A", file_name=path)


def get_list_of_impacted_files(repo_dir: str) -> Sequence[str]:
    """
    Return the full list of impacted files, either in the index or the working tree,
    combined such that deletion and renamings are ignored properly.
    """
    files = set()
    for diff in get_index_diff_file_names(repo_dir):
        if diff.change_type not in {"D", "R"}:
            files.add(diff.file_name)
    for diff in get_working_diff_file_names(repo_dir):
        if diff.change_type != "D":
            files.add(diff.file_name)
        else:
            files.remove(diff.file_name)
    return list(files)


def _run_command(command: str):
    os.system(command)


def _is_correct_path(path: str):
    return os.path.isdir(path) or path.endswith(".py")


def run_black_on(paths: Sequence[str], check_only: bool):
    options = "--check" if check_only else ""
    for path in paths:
        if _is_correct_path(path):
            _run_command(f"black --quiet {options} {path}")


def run_sort_include_on(paths: Sequence[str], check_only: bool):
    options = "-c" if check_only else ""
    for path in paths:
        if os.path.isdir(path):
            _run_command(f"isort {options} -sp {path}")
        elif path.endswith(".py"):
            _run_command(f"isort {options} {path}")


def run_flake8_on(paths: Sequence[str]):
    for path in paths:
        if _is_correct_path(path):
            _run_command(
                f"flake8 --max-line-length 88 --ignore E501,E203,E266,W503,E741 {path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", action="store_const", const=True, default=False)
    parser.add_argument(
        "-b", "--black", action="store_const", const=True, default=False
    )
    parser.add_argument(
        "-f", "--flake", action="store_const", const=True, default=False
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_const",
        const=True,
        default=False,
        help="Only check (no correction)",
    )
    parser.add_argument("-s", "--sort", action="store_const", const=True, default=False)
    parser.add_argument(
        "-r",
        "--repo",
        action="store_const",
        const=True,
        default=False,
        help="Running the linter on the whole repository",
    )
    args = parser.parse_args()

    if not args.repo:
        impacted_paths = get_list_of_impacted_files(os.getcwd())
    else:
        impacted_paths = ["vissl", "extra_scripts", "tools"]

    if args.all or args.sort:
        run_sort_include_on(impacted_paths, check_only=args.check)
    if args.all or args.black:
        run_black_on(impacted_paths, check_only=args.check)
    if args.all or args.flake:
        run_flake8_on(impacted_paths)
