from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from integrations.sandboxes.runtime.image import container_rootfs_tar_filter


class ContainerRootfsTarFilterTests(unittest.TestCase):
    def test_absolute_container_symlink_is_rewritten_inside_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "rootfs.tar"
            extract_path = root / "extract"
            extract_path.mkdir()

            payload = b"hello\n"
            target = tarfile.TarInfo("usr/bin/tool")
            target.size = len(payload)
            link = tarfile.TarInfo("bin/tool")
            link.type = tarfile.SYMTYPE
            link.linkname = "/usr/bin/tool"

            with tarfile.open(archive_path, "w") as archive:
                archive.addfile(target, io.BytesIO(payload))
                archive.addfile(link)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extract_path, filter=container_rootfs_tar_filter)

            self.assertEqual(os.readlink(extract_path / "bin" / "tool"), "../usr/bin/tool")
            self.assertEqual((extract_path / "bin" / "tool").read_bytes(), payload)

    def test_container_directory_mode_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "rootfs.tar"
            extract_path = root / "extract"
            extract_path.mkdir()

            tmp = tarfile.TarInfo("tmp")
            tmp.type = tarfile.DIRTYPE
            tmp.mode = 0o1777
            with tarfile.open(archive_path, "w") as archive:
                archive.addfile(tmp)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extract_path, filter=container_rootfs_tar_filter)

            self.assertEqual((extract_path / "tmp").stat().st_mode & 0o7777, 0o1777)

    def test_parent_traversal_is_still_rejected(self) -> None:
        member = tarfile.TarInfo("../../outside")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(tarfile.OutsideDestinationError):
                container_rootfs_tar_filter(member, temp_dir)


if __name__ == "__main__":
    unittest.main()
