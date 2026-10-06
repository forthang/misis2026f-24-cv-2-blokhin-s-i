#!/usr/bin/env python3
"""Turn Walnut projections and reconstruction slices into fixed-duration MP4s.

Export three projection rotations and four reconstruction slice sequences.
Projection orbits use individual fixed intensity windows; all reconstructions
share one fixed window so their brightness can be compared directly.
Original TIFFs are read only. No reconstruction or segmentation is performed.
Use --mosaic to also create a labelled 1920x1080 overview of all seven videos.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
MOSAIC_SERIES = ("tubeV1", "tubeV2", "tubeV3", "fdk_pos1", "fdk_pos2", "fdk_pos3", "full_AGD_50")


def mosaic_layout():
    """Three projection panels above four reconstruction panels, in pixels."""
    return [(i * 640, 0, 640, 540) for i in range(3)] + [
        (i * 480, 540, 480, 540) for i in range(4)
    ]


def make_mosaic_labels(path, videos):
    """Pillow labels avoid depending on an FFmpeg build with drawtext."""
    labels = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(labels)
    title_font = ImageFont.load_default(size=24)
    small_font = ImageFont.load_default(size=16)
    for index, (video, (x, y, width, height)) in enumerate(zip(videos, mosaic_layout())):
        color = (125, 210, 235) if index < 3 else (245, 205, 125)
        draw.text((x + 16, y + 9), video["source_series"], fill=color, font=title_font)
        kind = "Projection" if index < 3 else "Reconstruction"
        draw.text((x + 16, y + 38), f'{kind} / {video["input_frames"]} source frames',
                  fill=(200, 205, 215), font=small_font)
        if x:
            draw.line((x, y, x, y + height - 1), fill=(60, 65, 75), width=2)
    draw.line((0, 540, 1919, 540), fill=(60, 65, 75), width=2)
    draw.text((16, 1058), "Top: viewing angle. Bottom: slice depth. Shared playback progress, different physical coordinates.",
              fill=(180, 185, 195), font=small_font)
    labels.save(path)


def make_mosaic(videos, output_dir, output, duration, ffmpeg, ffprobe, overwrite):
    """Compose complete movies at the highest source rate without cropping."""
    if tuple(video["source_series"] for video in videos) != MOSAIC_SERIES:
        raise ValueError("The overview requires all seven series in the standard order")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    fps = max(Fraction(video["fps"]) for video in videos)
    frame_count = Fraction(str(duration)) * fps
    if frame_count.denominator != 1:
        raise ValueError("Overview duration and frame rate must define a whole number of frames")
    frame_count = int(frame_count)
    rate = f"{fps.numerator}/{fps.denominator}"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"{output.stem}: seven panels, 1920x1080, {rate} fps, {duration:g} s", flush=True)
    with tempfile.TemporaryDirectory(prefix=".mosaic-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        labels = temporary / "labels.png"
        encoded = temporary / "overview.mp4"
        make_mosaic_labels(labels, videos)
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "2"]
        filters = []
        layout = mosaic_layout()
        for i, (video, (_, _, width, height)) in enumerate(zip(videos, layout)):
            command += ["-i", str(output_dir / video["filename"])]
            filters.append(
                f"[{i}:v]setpts=PTS-STARTPTS,fps={rate},"
                f"scale={width - 24}:450:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={width}:{height}:(ow-iw)/2:64+(450-ih)/2:color=black,"
                f"setsar=1[v{i}]"
            )
        command += ["-loop", "1", "-framerate", rate, "-i", str(labels)]
        positions = "|".join(f"{x}_{y}" for x, y, _, _ in layout)
        filters.append("".join(f"[v{i}]" for i in range(7)) +
                       f"xstack=inputs=7:layout={positions}:fill=black:shortest=1[grid]")
        filters.append("[grid][7:v]overlay=shortest=1,format=yuv420p[out]")
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                    "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-r", rate,
                    "-video_track_timescale", str(fps.numerator), "-movflags", "+faststart", str(encoded)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"FFmpeg could not compose the overview: {result.stderr}")
        metadata = probe_video(encoded, ffprobe)
        stream = metadata["streams"][0]
        if (int(stream["width"]), int(stream["height"])) != (1920, 1080):
            raise RuntimeError("Unexpected overview dimensions")
        if int(stream["nb_read_frames"]) != frame_count:
            raise RuntimeError("Overview frame count differs from the requested count")
        if abs(float(stream["duration"]) - duration) > 0.000001:
            raise RuntimeError("Unexpected overview duration")
        if Fraction(stream["avg_frame_rate"]) != fps:
            raise RuntimeError("Unexpected overview frame rate")
        if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
            raise RuntimeError("Unexpected overview encoding")
        encoded.replace(output)
    return {
        "filename": output.name,
        "source_videos": [video["filename"] for video in videos],
        "layout": [dict(series=name, x=x, y=y, width=w, height=h)
                   for name, (x, y, w, h) in zip(MOSAIC_SERIES, layout)],
        "duration_seconds": float(stream["duration"]),
        "fps": rate,
        "frames": frame_count,
        "note": "Shared playback progress only: projection angles are not slice coordinates. "
                "Reconstruction frames repeat at the higher projection frame rate. "
                "Panels are scaled to fit without cropping; source videos and TIFFs are unchanged.",
        "output_size_bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "probe": metadata,
    }


def read_frame(path):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale image: {path}")
    return array


def find_frames(folder, prefix="scan_", suffix=".tif"):
    pattern = re.compile(re.escape(prefix) + r"(\d{6})" + re.escape(suffix))
    numbered = []
    for path in folder.glob(f"{prefix}*{suffix}"):
        match = pattern.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    if not numbered:
        raise FileNotFoundError(f"No {prefix}NNNNNN{suffix} files in {folder}")
    indices = [number for number, _ in numbered]
    missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
    if missing:
        raise ValueError(f"Missing {prefix} frame indices in {folder}: {missing[:20]}")
    return [path for _, path in numbered]


def fixed_window(frame_groups):
    samples = []
    sampled_indices = {}
    for name, frames in frame_groups.items():
        indices = np.unique(np.linspace(0, len(frames) - 1, min(33, len(frames)), dtype=int))
        sampled_indices[name] = indices.tolist()
        samples.extend(read_frame(frames[int(i)])[::4, ::4].reshape(-1) for i in indices)
    values = np.concatenate(samples)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values in sampled TIFF data")
    low, high = np.percentile(values, [0.5, 99.5]).astype(float)
    if high <= low:
        raise ValueError("The sampled TIFF sequences have no usable intensity range")
    return {"low": low, "high": high, "percentiles": [0.5, 99.5],
            "sampled_frame_indices": sampled_indices}


def probe_video(path, ffprobe):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate,duration:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def make_video(frames, output, duration, ffmpeg, ffprobe, overwrite, window, series, kind):
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    first = read_frame(frames[0])
    height, width = first.shape
    low, high = window["low"], window["high"]
    fps = Fraction(len(frames), 1) / Fraction(str(duration))
    rate = f"{fps.numerator}/{fps.denominator}"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".encoding-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        encoded = temporary / "video.mp4"
        logfile = temporary / "ffmpeg.log"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "gray",
            "-video_size", f"{width}x{height}", "-framerate", rate,
            "-i", "pipe:0", "-an", "-frames:v", str(len(frames)),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", rate,
            "-video_track_timescale", str(fps.numerator),
            "-movflags", "+faststart", str(encoded),
        ]
        print(f"{output.stem}: {len(frames)} frames, {width}×{height}, "
              f"{float(fps):.5g} fps, window [{low:.6g}, {high:.6g}]", flush=True)
        with logfile.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for i, path in enumerate(frames):
                    array = read_frame(path)
                    if array.shape != first.shape or not np.isfinite(array).all():
                        raise ValueError(f"Unexpected shape or nonfinite pixels: {path}")
                    display = (array.astype(np.float32) - low) * (255.0 / (high - low))
                    pixels = np.rint(np.clip(display, 0, 255)).astype(np.uint8)
                    process.stdin.write(pixels.tobytes())
                    if (i + 1) % 300 == 0 or i + 1 == len(frames):
                        print(f"  {i + 1}/{len(frames)}", flush=True)
                process.stdin.close()
                return_code = process.wait()
                if return_code:
                    raise RuntimeError(f"FFmpeg exited with {return_code}: {logfile.read_text()}")
            except BaseException as error:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if isinstance(error, BrokenPipeError):
                    raise RuntimeError(f"FFmpeg stopped reading frames: {logfile.read_text()}") from error
                raise
        metadata = probe_video(encoded, ffprobe)
        stream = metadata["streams"][0]
        actual_duration = float(stream["duration"])
        if int(stream["nb_read_frames"]) != len(frames):
            raise RuntimeError("Encoded frame count differs from the input sequence")
        if abs(actual_duration - duration) > 0.000001:
            raise RuntimeError(f"Unexpected duration: {actual_duration}, expected {duration}")
        if Fraction(stream["avg_frame_rate"]) != fps:
            raise RuntimeError("Encoded frame rate differs from the requested rate")
        if (int(stream["width"]), int(stream["height"])) != (width + width % 2, height + height % 2):
            raise RuntimeError("Unexpected encoded dimensions")
        if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
            raise RuntimeError("Unexpected video codec or pixel format")
        # Publish only the finished video after the entire stream has decoded.
        encoded.replace(output)
    return {
        "filename": output.name,
        "source_kind": kind,
        "source_series": series,
        "source_folder": str(frames[0].parent),
        "first_frame": frames[0].name,
        "last_frame": frames[-1].name,
        "input_frames": len(frames),
        "duration_seconds": actual_duration,
        "fps": rate,
        "frame_rate_decimal": float(fps),
        "display_window": window,
        "source_shape": [height, width],
        "source_dtype": str(first.dtype),
        "output_size_bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "probe": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walnut-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/walnut_videos")
    parser.add_argument("--kind", choices=("all", "projections", "reconstructions"), default="all",
                        help="Export all seven videos (default), only three projections, or four reconstructions")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration of each video in seconds")
    parser.add_argument("--mosaic", action="store_true",
                        help="Also create one 1920x1080 video showing all seven series at once (requires --kind all)")
    parser.add_argument("--overwrite", action="store_true", help="Replace videos created by an earlier run")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number")
    if args.mosaic and args.kind != "all":
        parser.error("--mosaic requires --kind all so all seven panels are present")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("Install FFmpeg (including ffprobe) and make both commands available in PATH")
    walnut = args.walnut_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    series = []
    if args.kind in ("all", "projections"):
        for i in (1, 2, 3):
            name = f"tubeV{i}"
            series.append((name, "projections", find_frames(walnut / "Projections" / name)))
    if args.kind in ("all", "reconstructions"):
        for name in ("fdk_pos1", "fdk_pos2", "fdk_pos3", "full_AGD_50"):
            frames = find_frames(walnut / "Reconstructions", prefix=f"{name}_", suffix=".tiff")
            series.append((name, "reconstructions", frames))
    tasks = [(name, kind, frames, output_dir / f"{walnut.name}_{name}.mp4")
             for name, kind, frames in series]
    for _, _, _, output in tasks:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    report_path = output_dir / "video_manifest.json"
    mosaic_path = output_dir / f"{walnut.name}_overview.mp4"
    if args.mosaic and mosaic_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {mosaic_path}; use --overwrite to replace it")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {report_path}; use --overwrite to replace it")
    reconstructions = {name: frames for name, kind, frames in series if kind == "reconstructions"}
    reconstruction_window = fixed_window(reconstructions) if reconstructions else None
    report = {
        "description": "Projection rotations and/or reconstruction slice sequences. Fixed window per projection orbit; shared window across all four reconstructions. Original orientation and all frames retained; odd dimensions padded on the right/bottom for H.264.",
        "kind": args.kind,
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "videos": [],
    }
    for name, kind, frames, output in tasks:
        window = reconstruction_window if kind == "reconstructions" else fixed_window({name: frames})
        result = make_video(frames, output, args.duration, ffmpeg, ffprobe, args.overwrite,
                            window, name, kind)
        report["videos"].append(result)
    if args.mosaic:
        report["mosaic"] = make_mosaic(report["videos"], output_dir, mosaic_path, args.duration,
                                        ffmpeg, ffprobe, args.overwrite)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
