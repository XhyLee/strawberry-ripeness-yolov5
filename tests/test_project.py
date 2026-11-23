"""Tests for dataset portability and preventing accidental COCO/class-order mixups."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import strawberry


class ProjectTests(unittest.TestCase):
    def test_yolo_directory_restores_cwd_after_failure(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary, patch.object(strawberry, "YOLO", Path(temporary)):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with strawberry.yolo_directory():
                    self.assertEqual(Path.cwd(), Path(temporary).resolve())
                    raise RuntimeError("inference failed")
        self.assertEqual(Path.cwd(), previous)

    def test_dataset_paths_are_relative_to_yaml_and_download_code_is_removed(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            for split in ("train", "val"):
                (root / "data" / "images" / split).mkdir(parents=True)
            config = root / "config" / "data.yaml"
            config.write_text(yaml.safe_dump({
                "path": "../data", "train": "images/train", "val": ["images/val"],
                "names": strawberry.CLASSES, "download": "raise RuntimeError('must not execute')",
            }), encoding="utf-8")
            with patch.object(strawberry, "ROOT", root):
                prepared = strawberry.prepare_data(config)
            data = yaml.safe_load(prepared.read_text())
            self.assertEqual(data["train"], str((root / "data/images/train").resolve()))
            self.assertEqual(data["val"], [str((root / "data/images/val").resolve())])
            self.assertNotIn("download", data)

    def test_wrong_class_order_and_missing_dataset_are_rejected(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "data.yaml"
            config.write_text(yaml.safe_dump({"names": strawberry.CLASSES[::-1]}))
            with self.assertRaisesRegex(ValueError, "class order"):
                strawberry.prepare_data(config)
            config.write_text(yaml.safe_dump({
                "names": strawberry.CLASSES, "train": "missing", "val": "missing",
            }))
            with self.assertRaisesRegex(ValueError, "not found"):
                strawberry.prepare_data(config)

    def test_missing_model_does_not_fall_back_to_coco_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "download-weights"):
                strawberry.local_weights(Path(temporary) / "missing.pt")

    def test_download_checksum_failure_does_not_install_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights").mkdir()
            (root / "weights/model.json").write_text(json.dumps({
                "sha256": hashlib.sha256(b"correct weights").hexdigest(),
                "download_url": "https://example.com/model.pt",
            }))
            destination = root / "weights/model.pt"
            with patch.object(strawberry, "ROOT", root), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(b"truncated weights")
            ):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    strawberry.download_weights(destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".download").exists())

    def test_invalid_threshold_is_rejected_before_loading_model(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            strawberry.main(["detect", "--source", "0", "--conf", "1.5"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
