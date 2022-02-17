from typing import Optional, Callable, List
import os
import zipfile
from PIL import Image


class ImagesArchiveDataset:
    def __init__(
        self,
        archive_dir: str,
        extensions: List[str] = ["jpg"],
        transform: Optional[Callable] = None,
    ):
        self.archive = zipfile.ZipFile(archive_dir, "r")
        self.filenames = self._get_filenames(self.archive, extensions)
        self.transform = transform

    def _get_filenames(self, archive, extensions):
        return [
            f for f in archive.namelist() if os.path.splitext(f)[1][1:] in extensions
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        entry = self.filenames[index]

        with self.archive.open(entry) as file:
            image = Image.open(file).convert("RGB")

        if self.transform is not None:
            return self.transform(image)

        return image
