"""Replay or verify the frozen UrbanTwin entry without redistributing challenge inputs."""
import argparse
import hashlib
from pathlib import Path
from subprocess import run
import sys
import zipfile

from src.common import digest, load_json, save_json


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=["v2x", "lumpi"])
    parser.add_argument("--kit", type=Path, help="Participant's original Codabench starting kit")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, help="Write fresh predictions here")
    parser.add_argument("--archive", type=Path, help="Check a privately held original signed ZIP")
    args = parser.parse_args()
    assets = ROOT / "assets" / args.track
    manifest = load_json(assets / "release.json")

    for name, expected in manifest["artifacts"].items():
        if digest(assets / name) != expected:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    print(f"{args.track}: frozen models and all 50 realism inputs verified", flush=True)

    if args.archive:
        release = manifest["signed_archive"]
        if digest(args.archive) != release["zip_sha256"]:
            raise ValueError("Archive checksum differs from the frozen entry")
        with zipfile.ZipFile(args.archive) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive member failed CRC validation")
            for name, expected in release["member_sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Archive member checksum differs: {name}")
        print(f"{args.track}: original signed submission verified byte-for-byte")

    if args.output:
        if args.kit is None:
            parser.error("--output requires --kit")
        ids = args.kit / "detection_test_frame_ids.json"
        if digest(ids) != manifest["canonical_ids_sha256"]:
            raise ValueError("Canonical frame-ID file differs from the original test kit")
        run(
            [sys.executable, "-m", "src.detector", "infer", "--track", args.track,
             "--checkpoint", str(assets / "best.pt"), "--refiner", str(assets / "refiner.joblib"),
             "--frame-ids", str(ids.resolve()), "--points",
             str((args.kit / "detection_test_frames").resolve()),
             "--channels", "3", "--output", str(args.output.resolve()), "--device", args.device],
            check=True,
            cwd=ROOT,
        )
        predictions = load_json(args.output)
        expected_frames = load_json(ids)
        if set(predictions) != set(expected_frames):
            raise ValueError("Predictions do not cover exactly the canonical test IDs")
        count = sum(map(len, predictions.values()))
        exact = digest(args.output) == manifest["signed_archive"]["member_sha256"]["predictions.json"]
        print(f"Fresh replay: {len(predictions)} frames, {count} detections, original prediction SHA-256 match: {exact}")


if __name__ == "__main__":
    main()

