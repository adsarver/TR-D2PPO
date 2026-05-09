#!/usr/bin/env python3
"""Combine per-agent lap videos into a square 2x2 comparison video.

This script is intentionally streaming-only: it reads one frame from each
source video, composites the grid frame, writes it, and moves on. It never
loads whole videos into memory.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_AGENTS = [
    ('BC_LSTM', 'BC-LSTM'),
    ('D2PPO', 'D2PPO'),
    ('GFPP', 'GFPP'),
    ('MPC', 'MPC'),
]


POSITIONS = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
]


def _parse_agents(agent_spec):
    if not agent_spec:
        return DEFAULT_AGENTS

    agents = []
    for item in agent_spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            key, label = item.split(':', 1)
            agents.append((key.strip(), label.strip()))
        else:
            agents.append((item, item.replace('_', '-')))
    return agents


def _parse_maps(map_spec):
    if not map_spec:
        return None
    return [name.strip() for name in map_spec.split(',') if name.strip()]


def _resolve_dir(path, sibling_name=None):
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute() or sibling_name is None:
        return candidate

    sibling = Path(__file__).resolve().parent / sibling_name
    return sibling if sibling.exists() else candidate


def _discover_maps(input_dir, agents):
    maps = set()
    for video_path in input_dir.glob('*.mp4'):
        stem = video_path.stem
        for agent_key, _ in agents:
            prefix = f'{agent_key}_'
            if stem.startswith(prefix) and len(stem) > len(prefix):
                maps.add(stem[len(prefix):])
                break
    return sorted(maps)


def _open_capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, None

    metadata = {
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    if metadata['frames'] <= 0 or metadata['fps'] <= 0:
        cap.release()
        return None, None
    return cap, metadata


def _fit_frame(frame, size, mode='cover', background=(18, 18, 18)):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return _blank_panel(size, background)

    if mode == 'contain':
        scale = min(size / width, size / height)
    else:
        scale = max(size / width, size / height)

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    if mode == 'contain':
        panel = _blank_panel(size, background)
        x0 = (size - resized_width) // 2
        y0 = (size - resized_height) // 2
        panel[y0:y0 + resized_height, x0:x0 + resized_width] = resized
        return panel

    x0 = max(0, (resized_width - size) // 2)
    y0 = max(0, (resized_height - size) // 2)
    return resized[y0:y0 + size, x0:x0 + size]


def _blank_panel(size, background=(18, 18, 18)):
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    panel[:] = background
    return panel


def _draw_label(panel, label):
    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, panel.shape[1] / 900.0)
    thickness = max(1, int(round(panel.shape[1] / 420.0)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad_x = int(round(14 * scale))
    pad_y = int(round(10 * scale))
    box_width = min(panel.shape[1] - 20, text_width + 2 * pad_x)
    box_height = text_height + baseline + 2 * pad_y

    overlay = panel.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height),
                  (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0, panel)
    cv2.rectangle(panel, (10, 10), (10 + box_width, 10 + box_height),
                  (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, label, (10 + pad_x, 10 + pad_y + text_height),
                font, scale, (25, 25, 25), thickness, cv2.LINE_AA)


def _make_canvas(panel_size, gap, background=(245, 245, 245)):
    canvas_size = 2 * panel_size + 3 * gap
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:] = background
    return canvas


def combine_map_videos(input_dir, output_dir, map_name, agents, output_size=1600,
                       fps=None, duration='longest', fit='cover', gap=12,
                       labels=True, allow_missing=False, frame_limit=None):
    panel_size = (output_size - 3 * gap) // 2
    panel_size -= panel_size % 2
    output_size = 2 * panel_size + 3 * gap

    sources = []
    missing = []
    for agent_key, label in agents:
        path = input_dir / f'{agent_key}_{map_name}.mp4'
        cap, metadata = _open_capture(path)
        if cap is None:
            missing.append(path)
        sources.append({
            'agent_key': agent_key,
            'label': label,
            'path': path,
            'cap': cap,
            'metadata': metadata,
            'last_frame': None,
        })

    valid_sources = [source for source in sources if source['cap'] is not None]
    if missing and not allow_missing:
        for path in missing:
            print(f'  [skip] {map_name}: missing or unreadable {path}')
        for source in valid_sources:
            source['cap'].release()
        return None
    if not valid_sources:
        print(f'  [skip] {map_name}: no readable input videos')
        return None

    source_fps = valid_sources[0]['metadata']['fps']
    writer_fps = float(fps) if fps else source_fps
    if duration == 'longest':
        output_frames = max(source['metadata']['frames'] for source in valid_sources)
    else:
        output_frames = min(source['metadata']['frames'] for source in valid_sources)
    if frame_limit is not None:
        output_frames = min(output_frames, int(frame_limit))
    if output_frames <= 0:
        print(f'  [skip] {map_name}: zero output frames')
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{map_name}_comparison_grid.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (output_size, output_size))
    if not writer.isOpened():
        for source in valid_sources:
            source['cap'].release()
        raise RuntimeError(f'Could not open video writer for {out_path}')

    print(f'  [combine] {map_name}: {output_frames} frames @ {writer_fps:.1f} FPS -> {out_path}')
    try:
        for frame_idx in range(output_frames):
            canvas = _make_canvas(panel_size, gap)
            for panel_idx, source in enumerate(sources[:4]):
                grid_x, grid_y = POSITIONS[panel_idx]
                x0 = gap + grid_x * (panel_size + gap)
                y0 = gap + grid_y * (panel_size + gap)

                frame = None
                cap = source['cap']
                if cap is not None:
                    ok, read_frame = cap.read()
                    if ok:
                        source['last_frame'] = read_frame
                        frame = read_frame
                    elif duration == 'longest' and source['last_frame'] is not None:
                        frame = source['last_frame']

                if frame is None:
                    panel = _blank_panel(panel_size)
                    if source['cap'] is None:
                        _draw_label(panel, f"{source['label']} unavailable")
                    else:
                        _draw_label(panel, f"{source['label']} ended")
                else:
                    panel = _fit_frame(frame, panel_size, mode=fit)
                    if labels:
                        _draw_label(panel, source['label'])

                canvas[y0:y0 + panel_size, x0:x0 + panel_size] = panel

            writer.write(canvas)
            if (frame_idx + 1) % 300 == 0:
                print(f'    {frame_idx + 1}/{output_frames} frames')
    finally:
        writer.release()
        for source in sources:
            if source['cap'] is not None:
                source['cap'].release()

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Combine per-agent videos from analysis/videos into 2x2 comparison videos.')
    parser.add_argument('--input-dir', default='analysis/videos',
                        help='Directory containing {agent}_{map}.mp4 files.')
    parser.add_argument('--output-dir', default='analysis/videos/combined',
                        help='Directory for combined comparison videos.')
    parser.add_argument('--maps', default=None,
                        help='Comma-separated maps to combine. Default: discover from input dir.')
    parser.add_argument('--agents', default=None,
                        help='Comma-separated agent prefixes, optionally prefix:Label. '
                             'Default: BC_LSTM,D2PPO,GFPP,MPC.')
    parser.add_argument('--size', type=int, default=1600,
                        help='Square output size in pixels.')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS. Default: first readable source FPS.')
    parser.add_argument('--duration', choices=('shortest', 'longest'), default='longest',
                        help='Default: longest, freezing ended panels on their final frame. Use shortest to stop at the shortest source.')
    parser.add_argument('--fit', choices=('cover', 'contain'), default='cover',
                        help='cover fills each square panel by cropping; contain letterboxes.')
    parser.add_argument('--gap', type=int, default=12,
                        help='Pixel gap around and between panels.')
    parser.add_argument('--frame-limit', type=int, default=None,
                        help='Optional cap for quick previews.')
    parser.add_argument('--allow-missing', action='store_true',
                        help='Render blank panels for missing/unreadable videos instead of skipping the map.')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not draw panel labels on top of source videos.')
    args = parser.parse_args()

    input_dir = _resolve_dir(args.input_dir, 'videos')
    output_dir = Path(args.output_dir)
    agents = _parse_agents(args.agents)
    map_names = _parse_maps(args.maps) or _discover_maps(input_dir, agents)

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    if not map_names:
        print(f'No matching videos found in {input_dir}')
        return

    print(f'Combining maps from {input_dir}: {", ".join(map_names)}')
    written = []
    for map_name in map_names:
        out_path = combine_map_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            map_name=map_name,
            agents=agents,
            output_size=args.size,
            fps=args.fps,
            duration=args.duration,
            fit=args.fit,
            gap=args.gap,
            labels=not args.no_labels,
            allow_missing=args.allow_missing,
            frame_limit=args.frame_limit,
        )
        if out_path is not None:
            written.append(out_path)

    if written:
        print('\nWrote:')
        for path in written:
            print(f'  {path}')
    else:
        print('\nNo combined videos were written.')


if __name__ == '__main__':
    main()
