# coding: utf-8

import os
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser, Namespace

DATASET_ROOT = Path(__file__).resolve().parent
DATA_ROOT = DATASET_ROOT.parent / 'data'

with open(DATA_ROOT / 'char_3755.txt', 'r', encoding='utf-8') as f:
    ALPHABET = f.read().strip()

with open(DATA_ROOT / 'decompose_27533_benchmark.txt', 'r', encoding='utf-8') as f:
    DECOMPOSE = {}
    for line in f.readlines():
        c, decomp = line.strip().split(':')
        DECOMPOSE[c] = decomp.split()

# split name -> [begin, end) range of class indices in ALPHABET (GB2312 order)
CHAR_SPLITS = {
    'seen': (0, 2755),
    'unseen': (2755, 3755),
    'seen-500': (0, 500),
    'seen-1000': (0, 1000),
    'seen-1500': (0, 1500),
    'seen-2000': (0, 2000),
    'train': (0, 3755),
    'mini': (0, 2755),    # first writer only
    'other': (-1, 0),     # characters outside ALPHABET
}
PAPER_CHAR_SPLITS = ['seen', 'unseen', 'seen-500', 'seen-1000', 'seen-1500', 'seen-2000']
COMP_THRESHOLDS = [50, 40, 30, 20, 10]


class DatasetGenerator:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.image_root_path = path / 'src' / 'image'
        self.df = self.get_dataset_df()

    def generate_char_dataset(self, splits: list[str]) -> None:
        for name in tqdm(splits, desc="Generating Character datasets"):
            begin, end = CHAR_SPLITS[name]
            result_path = self.path / name
            result_path.mkdir(exist_ok=True)

            df = self.df[(self.df['label_id'] >= begin) & (self.df['label_id'] < end)]
            if name == 'mini':
                df = df[df['style'] == df['style'].min()]

            label_set = set(df['label'].unique())
            print(f"Dataset '{name}' contains {len(label_set)} unique characters.")

            self.dataframe_to_csv(df, result_path)

    def generate_comp_dataset(self, thresholds: list[int]) -> None:
        df = self.df[self.df['label_id'] >= 0]
        df = self.add_minimum_component_frequency(df)

        for threshold in tqdm(thresholds, desc="Generating Component datasets"):
            seen_result_path = self.path / f'comp-{threshold}' / 'seen'
            unseen_result_path = self.path / f'comp-{threshold}' / 'unseen'

            seen_result_path.mkdir(exist_ok=True, parents=True)
            unseen_result_path.mkdir(exist_ok=True, parents=True)

            mask = df['min_comp_freq'] >= threshold

            self.dataframe_to_csv(df[mask], seen_result_path)
            self.dataframe_to_csv(df[~mask], unseen_result_path)

    def add_minimum_component_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        comp_freq = defaultdict(int)
        for c in ALPHABET:
            for dc in DECOMPOSE.get(c, []):
                comp_freq[dc] += 1

        min_freq_per_char = {}
        for c in ALPHABET:
            min_freq = min([comp_freq[dc] for dc in DECOMPOSE.get(c, [])])
            min_freq_per_char[c] = min_freq

            assert min_freq > 0, f"Character {c} has a component with zero frequency."

        df = df.copy()
        df['min_comp_freq'] = df['label'].map(min_freq_per_char)

        return df

    def get_dataset_df(self) -> pd.DataFrame:
        records = []
        image_root_list = sorted(self.image_root_path.iterdir())
        for image_root in tqdm(image_root_list, desc=f"Collecting {self.path.name}"):
            for image_file in image_root.iterdir():
                # file name: <label>-<sample index>.png (see gnt_parser.py)
                label, sample_id = image_file.stem.split('-')
                records.append({
                    'image': os.path.abspath(image_file),
                    'label': label,
                    'label_id': ALPHABET.find(label),
                    'style': int(image_root.name),
                    'sample_id': int(sample_id),
                })
        return pd.DataFrame.from_records(records)

    def dataframe_to_csv(self, df: pd.DataFrame, path: Path) -> None:
        # image paths are stored relative to the split directory so the dataset can be moved
        df = df.sort_values(by=['style', 'label_id', 'sample_id'])
        image = df['image'].apply(
            lambda image: os.path.relpath(image, os.path.abspath(path)).replace(os.sep, '/')
        )
        pd.DataFrame({'image': image, 'label': df['label']}).to_csv(
            str(path / 'labels.csv'), index=False
        )


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--path', '-p', action='append', type=Path,
        help='dataset root containing src/image/ (default: HWDB0, HWDB1, ICDAR next to this script)'
    )
    parser.add_argument(
        '--char-splits', nargs='*', default=PAPER_CHAR_SPLITS, choices=list(CHAR_SPLITS),
        help='character zero-shot splits to generate'
    )
    parser.add_argument(
        '--comp-thresholds', nargs='*', type=int, default=COMP_THRESHOLDS,
        help='radical zero-shot thresholds to generate; pass no value to skip'
    )

    args = parser.parse_args()
    if args.path is None:
        args.path = [DATASET_ROOT / name for name in ['HWDB0', 'HWDB1', 'ICDAR']]

    return args


def main() -> None:
    args = parse_argument()

    for p in args.path:
        generator = DatasetGenerator(p)
        generator.generate_char_dataset(args.char_splits)
        generator.generate_comp_dataset(args.comp_thresholds)


if __name__ == '__main__':
    main()
