#!/usr/bin/env python3
"""
video_creator.py — Render a bird's-eye race video for the first complete lap.

Re-uses the visual style from ``plot_overtake_snapshots`` in post_process.py
(map background, raceline, ego/opponent markers, loss-function threshold
circles, trajectory trail, scale bar) but produces a smooth mp4 video that
follows the ego car around the track for one full lap.
"""

import os
import pickle
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # headless backend for video
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.animation as animation
import yaml
from PIL import Image


# ───────────────────── tunables ──────────────────────
DT            = 0.01        # simulator timestep  (seconds / obs)
ZOOM_RADIUS   = 6.0         # half-width of the viewport (metres)
TRAIL_STEPS   = 300         # how many obs steps of past trajectory to show
LOOKAHEAD_STEPS = 30        # future trajectory hint (faded)
FPS           = 60           # output video frame rate
FRAME_SKIP    = 2            # render every N-th obs  (skip=2 → 50 Hz sim, 50 fps effective)

# Loss-function distance thresholds (from supervised_agent.py)
NEAR_THRESH   = 0.5         # metres – full avoidance weight
FAR_THRESH    = 1.0         # metres – transition to pure pursuit

# Car palette
EGO_COLOUR    = '#FF9800'   # orange
OPP_COLOURS   = ['#E53935', '#1E88E5', '#43A047',
                 '#8E24AA', '#00ACC1', '#FDD835',
                 '#6D4C41', '#546E7A']        # up to 8 opponents

RACELINE_COLOUR = '#AB47BC'
TRAIL_COLOUR    = '#2196F3'
FUTURE_COLOUR   = '#4CAF50'


# ───────────────────── helpers ──────────────────────

def _load_map(map_name, map_dir='maps'):
    """Return (img_array, extent, raceline_xy | None)."""
    img = Image.open(os.path.join(map_dir, map_name, f'{map_name}_map.png'))
    img_arr = np.array(img)
    with open(os.path.join(map_dir, map_name, f'{map_name}_map.yaml'), 'r') as f:
        meta = yaml.safe_load(f)
    origin = meta['origin']
    res    = meta['resolution']
    w, h   = img.size
    extent = [origin[0], origin[0] + w * res,
              origin[1], origin[1] + h * res]

    raceline_xy = None
    rl_path = os.path.join(map_dir, map_name, f'{map_name}_raceline.csv')
    if os.path.exists(rl_path):
        rl = np.genfromtxt(rl_path, delimiter=';', comments='#')
        raceline_xy = rl[:, 1:3]

    return img_arr, extent, raceline_xy


def _find_first_lap_range(obss, ego=0):
    """Return (start, end) obs indices spanning the first complete lap.

    We detect the lap boundary via the ``lap_counts`` field.  If unavailable
    we fall back to the full observation list.
    """
    if 'lap_counts' not in obss[0]:
        return 0, len(obss)

    start = 0
    initial_lap = int(obss[0]['lap_counts'][ego])
    for i in range(1, len(obss)):
        cur = int(obss[i]['lap_counts'][ego])
        if cur > initial_lap:
            return start, i
    # Lap never completed — use everything
    return 0, len(obss)


# ───────────────────── main renderer ──────────────────────

def render_lap_video(obss, map_name, out_path,
                     ego=0,
                     agent_name='Agent',
                     zoom_radius=ZOOM_RADIUS,
                     trail_steps=TRAIL_STEPS,
                     lookahead_steps=LOOKAHEAD_STEPS,
                     fps=FPS,
                     frame_skip=FRAME_SKIP,
                     map_dir='maps'):
    """Render the first lap of *obss* to an mp4 video.

    Parameters
    ----------
    obss : list[dict]
        List of observation dicts (poses_x, poses_y, poses_theta, collisions, …).
    map_name : str
        Track name used to load map image / raceline.
    out_path : str
        Destination file (.mp4).
    ego : int
        Index of the ego agent inside the observation arrays.
    agent_name : str
        Display name for the agent (shown in title).
    """
    img_arr, extent, raceline_xy = _load_map(map_name, map_dir)

    lap_start, lap_end = _find_first_lap_range(obss, ego)
    obss_lap = obss[lap_start:lap_end]
    if not obss_lap:
        print(f'  [video] No obs for {map_name} — skipping.')
        return

    n_obs  = len(obss_lap)
    n_cars = len(obss_lap[0]['poses_x'])

    # Pre-extract arrays for speed
    all_px = np.array([o['poses_x'] for o in obss_lap])   # (n_obs, n_cars)
    all_py = np.array([o['poses_y'] for o in obss_lap])
    all_th = np.array([o['poses_theta'] for o in obss_lap])
    all_col = np.array([o['collisions'] for o in obss_lap])

    # Frame indices (subsampled)
    frame_indices = np.arange(0, n_obs, frame_skip)
    n_frames = len(frame_indices)
    print(f'  [video] {map_name}: {n_obs} obs → {n_frames} frames '
          f'({n_obs * DT:.1f}s, {fps} fps output)')

    # ── Set up figure (16:9) ──
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Static layers drawn once
    ax.imshow(img_arr, extent=extent, aspect='equal', cmap='gray',
              origin='upper', zorder=0)
    if raceline_xy is not None:
        ax.plot(raceline_xy[:, 0], raceline_xy[:, 1],
                color=RACELINE_COLOUR, linewidth=1.2, alpha=0.45,
                linestyle='-', zorder=1)

    # Dynamic artists
    trail_line, = ax.plot([], [], color=TRAIL_COLOUR, linewidth=1.6,
                          linestyle='--', alpha=0.70, zorder=10)
    future_line, = ax.plot([], [], color=FUTURE_COLOUR, linewidth=2.0,
                           alpha=0.50, zorder=10)
    arrow_head, = ax.plot([], [], marker=(3, 0, 0), color=FUTURE_COLOUR,
                          markersize=12, alpha=0.70, zorder=11,
                          linestyle='None', markeredgecolor='black',
                          markeredgewidth=0.6)

    ego_dot, = ax.plot([], [], 'o', color=EGO_COLOUR, markersize=12,
                       markeredgecolor='black', markeredgewidth=1.2,
                       zorder=15)

    opp_dots = []
    opp_circles_near = []
    opp_circles_far  = []
    for j in range(n_cars):
        if j == ego:
            opp_dots.append(None)
            opp_circles_near.append(None)
            opp_circles_far.append(None)
            continue
        c_idx = (j - (1 if j > ego else 0)) % len(OPP_COLOURS)
        dot, = ax.plot([], [], 's', color=OPP_COLOURS[c_idx], markersize=12,
                       markeredgecolor='black', markeredgewidth=1.0,
                       zorder=13)
        opp_dots.append(dot)

        c_far = mpatches.Circle((0, 0), radius=FAR_THRESH,
                                fill=True, facecolor='#FFF3E0', alpha=0.30,
                                edgecolor='#FF9800', linewidth=2.0,
                                linestyle='--', zorder=8, visible=False)
        ax.add_patch(c_far)
        opp_circles_far.append(c_far)

        c_near = mpatches.Circle((0, 0), radius=NEAR_THRESH,
                                 fill=True, facecolor='#FFCDD2', alpha=0.40,
                                 edgecolor='#D32F2F', linewidth=2.0,
                                 linestyle='-', zorder=9, visible=False)
        ax.add_patch(c_near)
        opp_circles_near.append(c_near)

    # Scale bar (redrawn each frame because viewport moves)
    bar_line, = ax.plot([], [], color='white', linewidth=2.5,
                        solid_capstyle='butt', zorder=20)
    bar_text = ax.text(0, 0, '', color='white', fontsize=11,
                       ha='center', va='bottom', fontweight='bold', zorder=20,
                       path_effects=[
                           patheffects.withStroke(linewidth=2, foreground='black')
                       ])

    # Timer overlay
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        color='white', fontsize=13, fontweight='bold',
                        va='top', ha='left', zorder=25,
                        path_effects=[
                            patheffects.withStroke(linewidth=2.5, foreground='black')
                        ])

    ax.set_aspect('equal')
    ax.axis('off')
    title = ax.text(0.5, 0.97, '', transform=ax.transAxes,
                    color='white', fontsize=16, fontweight='bold',
                    ha='center', va='top', zorder=25,
                    path_effects=[
                        patheffects.withStroke(linewidth=3, foreground='black')
                    ])

    # Proximity threshold for showing circles (only when close)
    CIRCLE_SHOW_DIST = FAR_THRESH * 3.0

    def _update(frame_num):
        idx = frame_indices[frame_num]

        ex = all_px[idx, ego]
        ey = all_py[idx, ego]

        # Camera follows ego (16:9 aspect)
        zr = zoom_radius
        aspect = 16.0 / 9.0
        ax.set_xlim(ex - zr * aspect, ex + zr * aspect)
        ax.set_ylim(ey - zr, ey + zr)

        # Past trail
        t0 = max(0, idx - trail_steps)
        trail_line.set_data(all_px[t0:idx + 1, ego],
                            all_py[t0:idx + 1, ego])

        # Future hint (arrow)
        t1 = min(n_obs, idx + lookahead_steps)
        future_line.set_data(all_px[idx:t1, ego],
                             all_py[idx:t1, ego])
        # Arrowhead at the tip of the future line
        if t1 - idx > 2:
            dx = all_px[t1 - 1, ego] - all_px[t1 - 2, ego]
            dy = all_py[t1 - 1, ego] - all_py[t1 - 2, ego]
            angle = math.degrees(math.atan2(dy, dx))
            arrow_head.set_data([all_px[t1 - 1, ego]],
                                [all_py[t1 - 1, ego]])
            arrow_head.set_marker((3, 0, angle - 90))
            arrow_head.set_visible(True)
        else:
            arrow_head.set_visible(False)

        # Ego marker
        ego_dot.set_data([ex], [ey])

        # Opponents
        for j in range(n_cars):
            if j == ego:
                continue
            ox = all_px[idx, j]
            oy = all_py[idx, j]
            opp_dots[j].set_data([ox], [oy])

            d = math.hypot(ex - ox, ey - oy)
            show = d < CIRCLE_SHOW_DIST
            opp_circles_far[j].center = (ox, oy)
            opp_circles_far[j].set_visible(show)
            opp_circles_near[j].center = (ox, oy)
            opp_circles_near[j].set_visible(show)

        # Scale bar (bottom-left of viewport)
        aspect = 16.0 / 9.0
        bar_len = 1.0
        bx0 = ex - zr * aspect + 0.10 * zr
        by0 = ey - zr + 0.08 * zr
        bar_line.set_data([bx0, bx0 + bar_len], [by0, by0])
        bar_text.set_position((bx0 + bar_len / 2, by0 + 0.06 * zr))
        bar_text.set_text(f'{bar_len:.0f} m')

        # Timer
        t_sec = idx * DT
        time_text.set_text(f't = {t_sec:.2f} s')
        title.set_text(f'{agent_name} — {map_name}')

        artists = [trail_line, future_line, arrow_head, ego_dot,
                   bar_line, bar_text, time_text, title]
        for j in range(n_cars):
            if j != ego:
                artists += [opp_dots[j], opp_circles_far[j],
                            opp_circles_near[j]]
        return artists

    ani = animation.FuncAnimation(fig, _update, frames=n_frames,
                                  blit=True, repeat=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=3000,
                                    extra_args=['-pix_fmt', 'yuv420p'])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f'  Saved {out_path}')


# ────────────── top-level wrapper for multiprocessing ──────────────

def _render_one_video(args_tuple):
    """Picklable top-level function for ProcessPoolExecutor."""
    obss, map_name, out_path, ego, agent_name, zoom, fps, skip = args_tuple
    render_lap_video(obss, map_name, out_path,
                     ego=ego, agent_name=agent_name,
                     zoom_radius=zoom, fps=fps, frame_skip=skip)
    return out_path


# ───────────────────── CLI entry point ──────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Create a bird\'s-eye lap video from pickled race data.')
    parser.add_argument('pkl', nargs='?',
                        default='analysis/race_data_OVERTAKE.pkl',
                        help='Path to the pickled race data file.')
    parser.add_argument('--agent', default=None,
                        help='Agent key to render (default: first found).')
    parser.add_argument('--map', default=None,
                        help='Map to render (default: all maps).')
    parser.add_argument('--ego', type=int, default=0,
                        help='Ego agent index inside the obs arrays.')
    parser.add_argument('--zoom', type=float, default=ZOOM_RADIUS,
                        help='Viewport half-width in metres.')
    parser.add_argument('--fps', type=int, default=FPS,
                        help='Output video frame rate.')
    parser.add_argument('--skip', type=int, default=FRAME_SKIP,
                        help='Render every N-th observation step.')
    parser.add_argument('--out-dir', default='analysis/videos',
                        help='Directory for output mp4 files.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count).')
    args = parser.parse_args()

    n_workers = args.workers or os.cpu_count() or 1

    data = pickle.load(open(args.pkl, 'rb'))

    # Normalise agent names (same as post_process.py)
    if 'SupervisedAgent' in data:
        data['TR Agent'] = data.pop('SupervisedAgent')

    agents = [args.agent] if args.agent else list(data.keys())

    # Collect all render jobs
    jobs = []
    for agent_key in agents:
        if agent_key not in data:
            print(f'Agent "{agent_key}" not found in data. '
                  f'Available: {list(data.keys())}')
            continue
        maps = data[agent_key]
        map_names = [args.map] if args.map else list(maps.keys())

        for map_name in map_names:
            if map_name not in maps or not maps[map_name]:
                print(f'  No data for {agent_key}/{map_name} — skipping.')
                continue
            obss = maps[map_name]
            safe_agent = agent_key.replace(' ', '_')
            out_path = os.path.join(args.out_dir,
                                    f'{safe_agent}_{map_name}.mp4')
            jobs.append((obss, map_name, out_path, args.ego, agent_key,
                         args.zoom, args.fps, args.skip))

    if not jobs:
        print('No videos to render.')
        return

    n = len(jobs)
    w = min(n_workers, n)
    print(f'Rendering {n} video(s) with {w} worker(s)...')

    if w > 1:
        with ProcessPoolExecutor(max_workers=w) as pool:
            futures = {pool.submit(_render_one_video, j): j[2] for j in jobs}
            for fut in as_completed(futures):
                try:
                    path = fut.result()
                    print(f'  Finished: {path}')
                except Exception as e:
                    print(f'  ERROR rendering {futures[fut]}: {e}')
    else:
        for j in jobs:
            _render_one_video(j)

    print('\nDone.')


if __name__ == '__main__':
    main()