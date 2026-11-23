"""Portable entry point for the six-class YOLOv5 strawberry project."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
YOLO = ROOT / "yolov5-master"
DEFAULT_WEIGHTS = ROOT / "weights" / "strawberry_best.pt"
CLASSES = ["Early-Turning", "Green", "Late-Turning", "Red", "Turning", "White"]
VERSION = "1.0.0"


@contextmanager
def yolo_directory():
    """YOLOv5 computes relative roots at import; Windows drives must match."""
    previous = Path.cwd()
    os.chdir(YOLO)
    try:
        yield
    finally:
        os.chdir(previous)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def local_weights(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Model not found: {path}\nRun: python strawberry.py download-weights")
    if path.suffix.lower() != ".pt":
        raise ValueError("This entry point supports trusted YOLOv5 .pt checkpoints only.")
    return path


def prepare_data(config):
    """Resolve dataset paths relative to the YAML, never the working directory."""
    import yaml

    config = Path(config).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"Dataset configuration not found: {config}")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")
    names = data.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(CLASSES))):
            raise ValueError("Class IDs must be the integers 0 through 5.")
        names = [names[i] for i in range(len(CLASSES))]
    if names != CLASSES or data.get("nc", len(CLASSES)) != len(CLASSES):
        raise ValueError(f"Dataset class order must match the checkpoint: {CLASSES}")
    base = Path(data.get("path") or ".").expanduser()
    if not base.is_absolute():
        base = config.parent / base
    base = base.resolve()
    resolved = {"path": str(base), "nc": len(CLASSES), "names": CLASSES}
    for split in ("train", "val", "test"):
        entries = data.get(split)
        if not entries:
            if split == "test":
                continue
            raise ValueError(f"Dataset configuration is missing '{split}'.")
        if not isinstance(entries, (str, list)) or (isinstance(entries, list) and not all(isinstance(x, str) for x in entries)):
            raise ValueError(f"Dataset '{split}' must be a path or list of paths.")
        paths = [(base / entry).resolve() for entry in (entries if isinstance(entries, list) else [entries])]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ValueError(f"Dataset '{split}' not found: {missing}. See docs/DATASET.md.")
        resolved[split] = [str(path) for path in paths] if isinstance(entries, list) else str(paths[0])
    # Deliberately retain only data fields, so a third-party YAML cannot run download scripts.
    output = ROOT / "runs" / "configs" / (config.stem + ".resolved.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return output


def download_weights(destination):
    metadata = json.loads((ROOT / "weights" / "model.json").read_text(encoding="utf-8"))
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == metadata["sha256"]:
            print(f"Model already verified: {destination}")
            return
        raise ValueError(f"Destination already exists with different content: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    try:
        request = urllib.request.Request(metadata["download_url"], headers={"User-Agent": "strawberry-ripeness-yolov5"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != metadata["sha256"]:
            raise ValueError("Model checksum mismatch; download discarded.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {destination}")


def parser():
    cli = argparse.ArgumentParser(description="YOLOv5 strawberry ripeness detection, training and validation")
    cli.add_argument("--version", action="version", version=VERSION)
    commands = cli.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-weights", help="download the released six-class checkpoint")
    download.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
    commands.add_parser("info", help="show class names and recorded training metrics")
    cameras = commands.add_parser("cameras", help="scan local camera indices")
    cameras.add_argument("--limit", type=positive_int, default=5)
    for command in ("detect", "train", "validate"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
        cmd.add_argument("--imgsz", type=positive_int, default=640)
        cmd.add_argument("--name", default="strawberry")
        cmd.add_argument("--project", type=Path, default=ROOT / "runs" / command)
        cmd.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
        if command == "detect":
            cmd.add_argument("--source", required=True, help="image, directory, video, camera index or RTSP URL")
            cmd.add_argument("--conf", type=probability, default=0.5)
            cmd.add_argument("--iou", type=probability, default=0.45)
            cmd.add_argument("--view", action="store_true", help="show an OpenCV window")
            cmd.add_argument("--save-txt", action="store_true", help="save YOLO boxes with confidence scores")
            cmd.add_argument("--save-csv", action="store_true", help="save per-object class and confidence CSV")
            cmd.add_argument("--no-save", action="store_true", help="do not save annotated images/videos")
        else:
            cmd.add_argument("--data", type=Path, default=ROOT / "configs" / "strawberry.yaml")
            cmd.add_argument("--batch-size", type=positive_int, default=8)
            cmd.add_argument("--workers", type=int, default=0, help="0 is recommended on Windows")
            if command == "train":
                cmd.add_argument("--epochs", type=positive_int, default=100)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "info":
            print((ROOT / "weights" / "model.json").read_text(encoding="utf-8"))
            return 0
        if args.command == "download-weights":
            download_weights(args.output)
            return 0
        if args.command == "cameras":
            import cv2

            for index in range(args.limit):
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                try:
                    if capture.isOpened():
                        print(f"Camera available: {index}")
                finally:
                    capture.release()
            return 0
        weights = local_weights(args.weights)
        if args.command != "detect":
            if args.workers < 0:
                raise ValueError("--workers must be non-negative")
            data = prepare_data(args.data)
        sys.path.insert(0, str(YOLO))
        os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        common = dict(weights=str(weights), device=args.device, imgsz=args.imgsz,
                      project=str(args.project.resolve()), name=args.name)
        if args.command == "detect":
            source = args.source
            if not source.isdecimal() and "://" not in source:
                source = str(Path(source).expanduser().resolve())
        with yolo_directory():
            if args.command == "detect":
                from detect import run

                run(
                    **{**common, "imgsz": (args.imgsz, args.imgsz)}, source=source,
                    data=str(ROOT / "configs" / "strawberry.yaml"), conf_thres=args.conf,
                    iou_thres=args.iou, view_img=args.view, save_txt=args.save_txt,
                    save_conf=args.save_txt, save_csv=args.save_csv, nosave=args.no_save,
                )
            elif args.command == "train":
                from train import run

                run(**common, data=str(data), epochs=args.epochs, batch_size=args.batch_size,
                    workers=args.workers, hyp=str(YOLO / "data/hyps/hyp.scratch-low.yaml"))
            else:
                from val import run

                run(**common, data=str(data), batch_size=args.batch_size, workers=args.workers, half=False)
    except (ValueError, OSError, ImportError) as exc:
        cli.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
