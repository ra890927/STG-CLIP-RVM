import re
import struct
import codecs
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace


class GntParser:
    def __init__(self, path: Path) -> None:
        self.gnt_root_path = path / 'src' / 'gnt'
        self.result_root_path = path / 'src' / 'image'
        self.result_root_path.mkdir(exist_ok=True)

    def parse(self) -> None:
        gnt_files = list(self.gnt_root_path.iterdir())
        for gnt_file in tqdm(gnt_files, desc=f"Parsing {self.gnt_root_path.name}"):
            self.extract_gnt_to_image(gnt_file)

    def extract_gnt_to_image(self, gnt_file: Path) -> None:
        label_counter = {}
        with codecs.open(gnt_file, 'rb') as gf:
            while True:
                # remain data is not enough
                left_cache = gf.read(4)
                if len(left_cache) < 4:
                    break

                sample_size = struct.unpack('I', left_cache)[0]
                label = str(gf.read(2), 'gbk')
                width = struct.unpack('H', gf.read(2))[0]
                height = struct.unpack('H', gf.read(2))[0]

                assert sample_size == width * height + 10

                pixel_data = gf.read(width * height)
                img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width)

                if not re.match(r'[\u4e00-\u9fa5]', label):
                    continue

                label_counter[label] = label_counter.get(label, 0) + 1
                idx = label_counter[label]

                img = Image.fromarray(img_array, 'L')

                gnt_id = re.match(r'C?(\d+)(-f)?-[ft].gnt', gnt_file.name).group(1)
                (self.result_root_path / gnt_id).mkdir(exist_ok=True)
                img.save(self.result_root_path / gnt_id / f"{label}-{idx}.png")


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('path', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    gnt_parser = GntParser(args.path)
    gnt_parser.parse()


if __name__ == '__main__':
    main()
