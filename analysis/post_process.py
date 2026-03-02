# -*- coding: utf-8 -*-
import pickle
import json
import numpy as np

# ── Maps to include in the combined (aggregate) lap-stats figure ──
COMBINED_MAPS = ['Nuerburgring', 'Sepang', 'IMS']
EXPECTED_LAPS = 3  # number of laps each race is run for

race_data = dict()
lstm = dict()
mpc = dict()

with open('analysis/race_data_race2.pkl', 'rb') as f:
    race_data = pickle.load(f)
# with open('analysis/race_data_paper1.pkl', 'rb') as f:
#     lstm = pickle.load(f)
# with open('analysis/race_data_gf_8.0ms.pkl', 'rb') as f:
#     gf = pickle.load(f)
# with open('analysis/race_data_MPC_variable_speed3.pkl', 'rb') as f:
#     mpc = pickle.load(f)

race_data['TR Agent'] = race_data['SupervisedAgent']
del race_data['SupervisedAgent']
# race_data['MPCAgent'] = mpc['MPCAgent_12.0']
# race_data['GapFollow'] = gf['GapFollow']
# del race_data['SupervisedAgent']
# # del race_data['PurePursuit']

# race_data['Ours']['Catalunya'] = [
#     lap for lap in race_data['Ours']['Catalunya']
#     if lap['lap_counts'][0] < 3
# ]


def _load_centerline(map_name):
    """Load centerline and return (xy array, cumulative arc-length array, total lap length)."""
    import glob
    # Try exact name first, then fall back to any *_centerline.csv in the dir
    candidates = [
        f'maps/{map_name}/{map_name}_centerline.csv',
    ]
    found = glob.glob(f'maps/{map_name}/*_centerline.csv')
    candidates.extend(found)
    cl = None
    for path in candidates:
        try:
            cl = np.genfromtxt(path, delimiter=',', comments='#')
            break
        except Exception:
            continue
    if cl is None:
        raise FileNotFoundError(f'No centerline found for {map_name}')
    xy = cl[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_lens = np.sqrt((diffs**2).sum(axis=1))
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    # Close the loop
    lap_length = cum_s[-1] + np.linalg.norm(xy[-1] - xy[0])
    return xy, cum_s, lap_length


def _project_progress(pos, cl_xy, cl_cum_s, lap_length):
    """Return normalised progress in [0,1] for a world position along the centerline."""
    dists = np.sqrt(((cl_xy - pos)**2).sum(axis=1))
    idx = int(np.argmin(dists))
    return float(cl_cum_s[idx] / lap_length)


def create_lap_comparison():
    # Structure per lap: Positions, Velocity, Time, Max Speed,
    #                    Collisions (int count), Progress (float 0-1, only for DNF)
    # NOTE: lap_counts in the pickle is unreliable (numpy aliasing from env reuse).
    #       We detect laps by watching for lap_time resets (drop > 0.5 s).
    lap_comparison = dict()
    centerlines = {}  # cache

    for agent, maps in race_data.items():
        for map_name, obss in maps.items():
            if map_name not in lap_comparison:
                lap_comparison[map_name] = dict()

            # Load centerline for progress computation
            if map_name not in centerlines:
                try:
                    centerlines[map_name] = _load_centerline(map_name)
                except Exception:
                    centerlines[map_name] = None

            laps = []
            current_lap = {'Positions': [], 'Velocity': [], 'Time': None,
                           'Collisions': 0, 'WallCollisions': 0, 'AgentCollisions': 0,
                           'WallColSteps': 0, 'AgentColSteps': 0}
            col_exit = False
            prev_lap_time = 0.0
            prev_collision = 0

            for obs in obss:
                if obs.get('col_exit', False):
                    current_lap['Time'] = 'DNF'
                    current_lap['DNF_reason'] = 'collision'
                    laps.append(current_lap)
                    col_exit = True
                    break

                lt = obs.get('lap_time', None)

                # Count collision onsets and total collision steps
                # collisions flag: 0 = none, 1 = wall, 2 = agent-agent
                cur_col = int(obs.get('collisions', [0])[0])
                in_collision = cur_col >= 1
                was_in_collision = prev_collision >= 1
                if cur_col == 1:
                    current_lap['WallColSteps'] += 1
                elif cur_col == 2:
                    current_lap['AgentColSteps'] += 1
                if in_collision and not was_in_collision:
                    current_lap['Collisions'] += 1
                    if cur_col == 1:
                        current_lap['WallCollisions'] += 1
                    elif cur_col == 2:
                        current_lap['AgentCollisions'] += 1
                prev_collision = cur_col

                # Detect a lap completion: lap_time resets (drops significantly)
                if lt is not None and prev_lap_time > 0.5 and lt < prev_lap_time - 0.5:
                    current_lap['Time'] = float(prev_lap_time)
                    laps.append(current_lap)
                    current_lap = {'Positions': [], 'Velocity': [], 'Time': None,
                                   'Collisions': 0, 'WallCollisions': 0, 'AgentCollisions': 0,
                                   'WallColSteps': 0, 'AgentColSteps': 0}

                # Append data to current lap
                current_lap['Positions'].append((obs['poses_x'][0], obs['poses_y'][0]))
                current_lap['Velocity'].append(obs['linear_vels_x'][0])

                if lt is not None:
                    prev_lap_time = float(lt)

            # Append last in-progress lap if it has data and wasn't already added
            if current_lap['Positions'] and (not laps or laps[-1] is not current_lap):
                if col_exit:
                    # Already marked 'DNF' above
                    laps.append(current_lap)
                elif prev_lap_time > 0.5:
                    # Sim ended normally (no collision).  The data-collection
                    # loop exits right after the final lap completes, so the
                    # trailing data IS the completed last lap.
                    current_lap['Time'] = float(prev_lap_time)
                    laps.append(current_lap)
                else:
                    # Trailing lap with virtually no time – truly incomplete
                    current_lap['Time'] = 'DNF'
                    current_lap['DNF_reason'] = 'incomplete'
                    laps.append(current_lap)

            # Compute per-lap metrics
            SIM_DT = 0.01  # f110_gym timestep
            for lap in laps:
                lap['Max Speed'] = float(np.max(lap['Velocity'])) if lap['Velocity'] else 0.0
                
                # Collision Severity Score (CSS)
                # Wall collisions (stuck/DNF-causing) weighted 3× more than
                # agent-agent bumps (typically brief and recoverable).
                # DNF laps get a flat penalty of 1.0 so that crashing out
                # always scores worse than completing a lap with bumps.
                # CSS = 0.5 * (weighted_event_rate + weighted_time_fraction) [+ DNF penalty]
                W_WALL  = 3.0
                W_AGENT = 1.0
                DNF_PENALTY = 1.0
                
                n_steps = len(lap['Positions'])
                lap_duration = n_steps * SIM_DT if n_steps > 0 else 1.0
                
                wall_events = lap.get('WallCollisions', 0)
                agent_events = lap.get('AgentCollisions', 0)
                wall_steps = lap.get('WallColSteps', 0)
                agent_steps = lap.get('AgentColSteps', 0)
                
                weighted_events = W_WALL * wall_events + W_AGENT * agent_events
                weighted_col_time = (W_WALL * wall_steps + W_AGENT * agent_steps) * SIM_DT
                
                event_rate = weighted_events / lap_duration
                time_frac  = weighted_col_time / lap_duration
                base_css = 0.5 * (event_rate + time_frac)
                
                # DNF penalty: only for collision-caused DNFs, not sim-ended incomplete laps
                is_collision_dnf = (lap['Time'] == 'DNF' and lap.get('DNF_reason') == 'collision')
                lap['CSS'] = base_css + (DNF_PENALTY if is_collision_dnf else 0.0)
                
                # Progress before failure (for DNF laps)
                if lap['Time'] == 'DNF' and lap['Positions'] and centerlines.get(map_name) is not None:
                    cl_xy, cl_cum_s, lap_length = centerlines[map_name]
                    # Use maximum progress seen during the lap (handles wrap-around
                    # near start/finish where final position could look like ~0)
                    positions = np.array(lap['Positions'])
                    prog_all = np.array([
                        _project_progress(positions[k], cl_xy, cl_cum_s, lap_length)
                        for k in range(len(positions))
                    ])
                    lap['Progress'] = float(np.max(prog_all))
                else:
                    lap['Progress'] = 1.0 if lap['Time'] != 'DNF' else 0.0

            lap_comparison[map_name][agent] = {
                'laps': laps,
                'col_exit': col_exit,
            }

    return lap_comparison

lap_comparison = create_lap_comparison()
def pretty_print_dict(d):
    print(json.dumps(d, indent=4))

def world_to_pixel(x, y, origin, resolution, img_height):
    """Convert world coordinates (meters) to image pixel coordinates.
    
    The map YAML origin is the world position of the bottom-left corner of the image.
    Image y increases downward, world y increases upward, so we flip the y axis.
    """
    px = (x - origin[0]) / resolution
    py = img_height - (y - origin[1]) / resolution
    return px, py

def plot_raceline_on_map_image(d):
    import matplotlib.pyplot as plt
    import matplotlib.collections as mcoll
    import yaml
    import os
    from PIL import Image

    V_MIN, V_MAX = 0.0, 15.0  # static colour-bar scale (m/s)
    N_INTERP = 2000
    DPI = 300

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        img = Image.open(f"maps/{map_name}/{map_name}_map.png")
        img_arr = np.array(img)
        img_width, img_height = img.size

        with open(f"maps/{map_name}/{map_name}_map.yaml", 'r') as f:
            map_yaml = yaml.safe_load(f)
        origin = map_yaml['origin']
        resolution = map_yaml['resolution']

        cmap = plt.cm.plasma
        norm = plt.Normalize(V_MIN, V_MAX)

        # ── One figure per agent ──
        for agent_name, data in agents.items():
            laps = data['laps']
            if not laps:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Compute world-coordinate extent of the map image
            x_min = origin[0]
            y_min = origin[1]
            x_max = origin[0] + img_width * resolution
            y_max = origin[1] + img_height * resolution
            # Show map in world coords; origin='upper' keeps image orientation correct
            ax.imshow(img_arr, extent=[x_min, x_max, y_min, y_max],
                      aspect='equal', cmap='gray', origin='upper')

            interp_px_all, interp_py_all, interp_v_all = [], [], []
            common_t = np.linspace(0, 1, N_INTERP)

            for lap in laps:
                if not lap['Positions'] or len(lap['Positions']) < 2:
                    continue
                is_collision_dnf = lap.get('DNF_reason') == 'collision'
                x_w, y_w = zip(*lap['Positions'])
                px = np.array(x_w)
                py = np.array(y_w)
                v = np.array(lap['Velocity'])

                # Coloured line segments
                points = np.array([px, py]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_v = (v[:-1] + v[1:]) / 2.0

                if is_collision_dnf:
                    # Draw collision laps in red so the crash location is visible
                    lc = mcoll.LineCollection(segments, colors='red',
                                              linewidths=1.0, alpha=0.6,
                                              linestyles='dashed')
                    ax.add_collection(lc)
                    # Mark crash point with an X
                    ax.plot(px[-1], py[-1], 'rx', markersize=8, markeredgewidth=2,
                            zorder=10)
                else:
                    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                              linewidths=0.8, alpha=0.55)
                    lc.set_array(seg_v)
                    ax.add_collection(lc)

                # Only include non-collision laps in the mean raceline
                if not is_collision_dnf:
                    t = np.linspace(0, 1, len(px))
                    interp_px_all.append(np.interp(common_t, t, px))
                    interp_py_all.append(np.interp(common_t, t, py))
                    interp_v_all.append(np.interp(common_t, t, v))

            # Mean raceline coloured by average speed
            if interp_px_all:
                mean_px = np.mean(interp_px_all, axis=0)
                mean_py = np.mean(interp_py_all, axis=0)
                mean_v  = np.mean(interp_v_all, axis=0)

                points = np.array([mean_px, mean_py]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_v = (mean_v[:-1] + mean_v[1:]) / 2.0

                lc_mean = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                               linewidths=1.6, zorder=4)
                lc_mean.set_array(seg_v)
                ax.add_collection(lc_mean)

            # Colour bar (static 0-15)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Speed (m/s)", fontsize=16)

            ax.axis('off')
            safe_agent = agent_name.replace(' ', '_')
            ax.set_title(f"{agent_name} — {map_name}  ({len(laps)} laps)",
                         fontsize=18, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
            fig.tight_layout()
            fig.savefig(f"{map_dir}/raceline_{safe_agent}.png",
                        bbox_inches='tight', dpi=DPI, facecolor='white')
            plt.close(fig)

# plot_raceline_on_map_image called after tables (see below)

def plot_velocity_profiles(d):
    import matplotlib.pyplot as plt
    import os

    agent_colors = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        for i, agent_name in enumerate(agents.keys()):
            if agent_name not in agent_colors:
                agent_colors[agent_name] = color_cycle[i % len(color_cycle)]

        # ── Per-agent velocity profile figures ──
        for agent_name, data in agents.items():
            n_laps = len(data['laps'])
            if n_laps == 0:
                continue

            n_plots = n_laps + 1  # +1 for average summary
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
            axes = np.array(axes).flatten()

            color = agent_colors[agent_name]

            for lap_idx in range(n_laps):
                ax = axes[lap_idx]
                lap = data['laps'][lap_idx]
                v = np.array(lap['Velocity'])
                if len(v) > 0:
                    grid = np.linspace(0, 1, len(v))
                    ax.scatter(grid, v, s=1.5, color=color)
                t = lap['Time']
                try:
                    t_str = f" — {float(t):.2f}s"
                except (TypeError, ValueError):
                    t_str = f" — {t}" if t is not None else ""
                ax.set_title(f"Lap {lap_idx + 1}{t_str}", fontsize=9)
                ax.set_xlabel("Normalised Lap Progress", fontsize=7)
                ax.set_ylabel("Speed (m/s)", fontsize=7)
                ax.grid(True, linewidth=0.4, alpha=0.5)

            # Summary subplot — average ± std across laps
            ax = axes[n_laps]
            common_grid = np.linspace(0, 1, 500)
            interp_laps = []
            for lap in data['laps']:
                v = np.array(lap['Velocity'])
                if len(v) > 1:
                    t = np.linspace(0, 1, len(v))
                    interp_laps.append(np.interp(common_grid, t, v))
            if interp_laps:
                mean_v = np.mean(interp_laps, axis=0)
                std_v = np.std(interp_laps, axis=0) if len(interp_laps) > 1 else np.zeros_like(mean_v)
                ax.plot(common_grid, mean_v, linewidth=1.2, color=color)
                ax.fill_between(common_grid, mean_v - std_v, mean_v + std_v,
                                color=color, alpha=0.15)
            ax.set_title("Average ± Std", fontsize=9)
            ax.set_xlabel("Normalised Lap Progress", fontsize=7)
            ax.set_ylabel("Speed (m/s)", fontsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.5)

            for i in range(n_laps + 1, len(axes)):
                axes[i].set_visible(False)

            safe_agent = agent_name.replace(' ', '_')
            fig.suptitle(f"Velocity Profile — {agent_name} — {map_name}", fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(f"{map_dir}/velocity_profile_{safe_agent}.png", dpi=300)
            plt.close(fig)

# plot_velocity_profiles called after tables (see below)

def plot_lap_stats_table(d):
    import matplotlib.pyplot as plt
    import os

    TARGET_LAPS = 10  # K for CR@K

    def style_table(table, n_cols, n_rows):
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(list(range(n_cols)))
        for col in range(n_cols):
            table[0, col].set_facecolor('#2c3e50')
            table[0, col].set_text_props(color='white', fontweight='bold')
        for row in range(1, n_rows + 1):
            color = '#ecf0f1' if row % 2 == 0 else 'white'
            for col in range(n_cols):
                table[row, col].set_facecolor(color)

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        agent_names = sorted(agents.keys())

        # ── Per-Lap Table (all laps, including DNF) ──
        per_lap_cols = ['Agent', 'Lap', 'Lap Time (s)', 'Max Speed (m/s)',
                        'Collisions', 'CSS', '% Completed', 'Progress @ Fail']
        per_lap_rows = []
        for agent_name in agent_names:
            data = agents[agent_name]
            laps = data['laps']
            for i, lap in enumerate(laps):
                t = lap['Time']
                try:
                    t_str = f"{float(t):.3f}"
                except (TypeError, ValueError):
                    t_str = 'DNF' if t == 'DNF' else ('—' if t is None else str(t))
                is_dnf = (t == 'DNF')
                spd = f"{lap['Max Speed']:.3f}"
                cols_str = str(lap.get('Collisions', 0))
                css_str = f"{lap.get('CSS', 0.0):.4f}"
                progress = lap.get('Progress', 1.0)
                pct_completed = f"{progress * 100:.1f}%"
                prog_at_fail = f"{progress:.3f}" if is_dnf else '—'
                per_lap_rows.append([
                    agent_name,
                    str(i + 1),
                    t_str,
                    spd,
                    cols_str,
                    css_str,
                    pct_completed,
                    prog_at_fail,
                ])

        # ── Summary Table (all agents, all laps included) ──
        n_all_laps = max(len(agents[a]['laps']) for a in agent_names) if agent_names else 0
        cr_k = n_all_laps if n_all_laps > 0 else TARGET_LAPS
        summary_cols = ['Agent', f'CR@{cr_k}', 'Lap Time (s)',
                        'Max Speed (m/s)', 'Collisions / Lap', 'CSS (↓)',
                        '% Completed (total)', 'Progress @ Failure']
        summary_rows = []
        for agent_name in agent_names:
            laps = agents[agent_name]['laps']
            n_total_laps = len(laps) if laps else 1

            # Completed lap times
            valid_times = []
            for lap in laps:
                try:
                    valid_times.append(float(lap['Time']))
                except (TypeError, ValueError):
                    pass
            n_completed = len(valid_times)

            # CR@K: 1 if completed >= K laps, else 0
            cr_at_k = '1' if n_completed >= cr_k else '0'

            # Lap time mean ± std (completed laps only)
            if valid_times:
                mean_t = np.mean(valid_times)
                std_t = np.std(valid_times) if len(valid_times) > 1 else 0.0
                time_str = f"{mean_t:.3f} ± {std_t:.3f}"
            else:
                time_str = 'DNF'

            # Max speed across ALL laps (including DNF)
            all_speeds = [lap['Max Speed'] for lap in laps if lap['Velocity']]
            max_speed = f"{np.max(all_speeds):.3f}" if all_speeds else '—'

            # Collision rate (collisions per lap, ALL laps)
            total_cols = sum(lap.get('Collisions', 0) for lap in laps)
            col_rate = f"{total_cols / n_total_laps:.2f}"

            # Mean Collision Severity Score across ALL laps
            css_vals = [lap.get('CSS', 0.0) for lap in laps]
            mean_css = np.mean(css_vals) if css_vals else 0.0
            css_str = f"{mean_css:.4f}"

            # % completed: sum lap progress / 3, scaled to 0-100%
            prog_vals_all = [lap.get('Progress', 1.0) for lap in laps]
            pct = np.sum(prog_vals_all) / 3.0 * 100.0 if prog_vals_all else 0.0
            pct_str = f"{pct:.1f}%"

            # Progress before failure (DNF laps only)
            dnf_laps = [lap for lap in laps if lap['Time'] == 'DNF']
            if dnf_laps:
                prog_vals = [lap.get('Progress', 0.0) for lap in dnf_laps]
                prog_str = f"{np.mean(prog_vals):.3f}"
            else:
                prog_str = '—'

            summary_rows.append([agent_name, cr_at_k, time_str,
                                 max_speed, col_rate, css_str, pct_str, prog_str])

        # ── Draw both tables ──
        n_per_lap = len(per_lap_rows)
        n_summary = len(summary_rows)
        if n_per_lap == 0:
            continue
        height = 0.45 * (n_per_lap + n_summary + 6)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, max(4, height)),
            gridspec_kw={'height_ratios': [max(1, n_per_lap + 1),
                                           max(1, n_summary + 1)]})

        # Per-lap table
        ax1.axis('off')
        ax1.set_title(f"Per-Lap Results — {map_name}", fontsize=11,
                      fontweight='bold', pad=10)
        t1 = ax1.table(cellText=per_lap_rows, colLabels=per_lap_cols,
                        cellLoc='center', loc='center')
        style_table(t1, len(per_lap_cols), n_per_lap)

        # Summary table
        ax2.axis('off')
        ax2.set_title("Summary", fontsize=11, fontweight='bold', pad=10)
        t2 = ax2.table(cellText=summary_rows, colLabels=summary_cols,
                        cellLoc='center', loc='center')
        style_table(t2, len(summary_cols), n_summary)

        fig.tight_layout()
        fig.savefig(f"{map_dir}/lap_stats.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

# ── Run tables first (fast), then plots (slow) ──
plot_lap_stats_table(lap_comparison)


def plot_combined_lap_stats(d, map_list):
    """Produce a single combined lap-stats figure that aggregates laps across
    all maps in *map_list*.  Per-lap rows show (Map, Lap, …) and the summary
    table shows one row per agent with statistics pooled over every map."""
    import matplotlib.pyplot as plt
    import os

    TARGET_LAPS = 10

    def style_table(table, n_cols, n_rows):
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(list(range(n_cols)))
        for col in range(n_cols):
            table[0, col].set_facecolor('#2c3e50')
            table[0, col].set_text_props(color='white', fontweight='bold')
        for row in range(1, n_rows + 1):
            color = '#ecf0f1' if row % 2 == 0 else 'white'
            for col in range(n_cols):
                table[row, col].set_facecolor(color)

    # Filter to only maps present in both the requested list and the data
    active_maps = [m for m in map_list if m in d]
    if not active_maps:
        print("  [combined] No matching maps found – skipping.")
        return

    # Collect the union of agent names across all selected maps
    all_agents = sorted({a for m in active_maps for a in d[m]})

    # ── Per-Lap Table ──
    per_lap_cols = ['Agent', 'Map', 'Lap', 'Lap Time (s)', 'Max Speed (m/s)',
                    'Collisions', 'CSS', '% Completed', 'Progress @ Fail']
    per_lap_rows = []
    # Also collect laps per agent for the summary
    agent_laps = {a: [] for a in all_agents}  # agent -> [lap_dict, …]

    for map_name in active_maps:
        agents = d[map_name]
        for agent_name in all_agents:
            if agent_name not in agents:
                continue
            data = agents[agent_name]
            laps = data['laps']
            for i, lap in enumerate(laps):
                agent_laps[agent_name].append(lap)
                t = lap['Time']
                try:
                    t_str = f"{float(t):.3f}"
                except (TypeError, ValueError):
                    t_str = 'DNF' if t == 'DNF' else ('—' if t is None else str(t))
                is_dnf = (t == 'DNF')
                spd = f"{lap['Max Speed']:.3f}"
                cols_str = str(lap.get('Collisions', 0))
                css_str = f"{lap.get('CSS', 0.0):.4f}"
                progress = lap.get('Progress', 1.0)
                pct_completed = f"{progress * 100:.1f}%"
                prog_at_fail = f"{progress:.3f}" if is_dnf else '—'
                per_lap_rows.append([
                    agent_name,
                    map_name,
                    str(i + 1),
                    t_str,
                    spd,
                    cols_str,
                    css_str,
                    pct_completed,
                    prog_at_fail,
                ])

    # ── Summary Table ──
    total_lap_counts = [len(agent_laps[a]) for a in all_agents]
    n_all_laps = max(total_lap_counts) if total_lap_counts else 0
    cr_k = n_all_laps if n_all_laps > 0 else TARGET_LAPS
    summary_cols = ['Agent', 'Maps', 'Total Laps', f'CR@{cr_k}',
                    'Lap Time (s)', 'Max Speed (m/s)',
                    'Collisions / Lap', 'CSS (↓)',
                    '% Completed (total)', 'Progress @ Failure']
    summary_rows = []

    for agent_name in all_agents:
        laps = agent_laps[agent_name]
        n_total = len(laps) if laps else 1
        n_maps = sum(1 for m in active_maps if agent_name in d[m])

        valid_times = []
        for lap in laps:
            try:
                valid_times.append(float(lap['Time']))
            except (TypeError, ValueError):
                pass
        n_completed = len(valid_times)

        cr_at_k = '1' if n_completed >= cr_k else '0'

        # Lap time: compute per-map mean, then average those means (± std across maps)
        per_map_mean_times = []
        for m in active_maps:
            if agent_name not in d[m]:
                continue
            map_times = []
            for lap in d[m][agent_name]['laps']:
                try:
                    map_times.append(float(lap['Time']))
                except (TypeError, ValueError):
                    pass
            if map_times:
                per_map_mean_times.append(np.mean(map_times))
        if per_map_mean_times:
            mean_t = np.mean(per_map_mean_times)
            std_t = np.std(per_map_mean_times) if len(per_map_mean_times) > 1 else 0.0
            time_str = f"{mean_t:.3f} ± {std_t:.3f}"
        else:
            time_str = 'DNF'

        all_speeds = [lap['Max Speed'] for lap in laps if lap.get('Velocity')]
        max_speed = f"{np.max(all_speeds):.3f}" if all_speeds else '—'

        total_cols = sum(lap.get('Collisions', 0) for lap in laps)
        col_rate = f"{total_cols / n_total:.2f}"

        # CSS: compute per-map mean, then average across maps
        per_map_css = []
        for m in active_maps:
            if agent_name not in d[m]:
                continue
            map_css = [lap.get('CSS', 0.0) for lap in d[m][agent_name]['laps']]
            if map_css:
                per_map_css.append(np.mean(map_css))
        mean_css = np.mean(per_map_css) if per_map_css else 0.0
        css_str = f"{mean_css:.4f}"

        # Per-map: sum lap progress / 3, then average across maps
        per_map_pcts = []
        for m in active_maps:
            if agent_name not in d[m]:
                continue
            map_laps = d[m][agent_name]['laps']
            map_prog = sum(lap.get('Progress', 1.0) for lap in map_laps)
            per_map_pcts.append(map_prog / 3.0 * 100.0)
        pct = np.mean(per_map_pcts) if per_map_pcts else 0.0
        pct_str = f"{pct:.1f}%"

        dnf_laps = [lap for lap in laps if lap['Time'] == 'DNF']
        if dnf_laps:
            prog_vals = [lap.get('Progress', 0.0) for lap in dnf_laps]
            prog_str = f"{np.mean(prog_vals):.3f}"
        else:
            prog_str = '—'

        summary_rows.append([agent_name, str(n_maps), str(len(laps)),
                             cr_at_k, time_str, max_speed, col_rate,
                             css_str, pct_str, prog_str])

    # ── Draw ──
    n_per_lap = len(per_lap_rows)
    n_summary = len(summary_rows)
    if n_per_lap == 0:
        return

    height = 0.45 * (n_per_lap + n_summary + 6)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, max(4, height)),
        gridspec_kw={'height_ratios': [max(1, n_per_lap + 1),
                                       max(1, n_summary + 1)]})

    ax1.axis('off')
    maps_label = ', '.join(active_maps[:5])
    if len(active_maps) > 5:
        maps_label += f' … ({len(active_maps)} maps total)'
    ax1.set_title(f"Combined Per-Lap Results — {maps_label}",
                  fontsize=11, fontweight='bold', pad=10)
    t1 = ax1.table(cellText=per_lap_rows, colLabels=per_lap_cols,
                    cellLoc='center', loc='center')
    style_table(t1, len(per_lap_cols), n_per_lap)

    ax2.axis('off')
    ax2.set_title("Combined Summary", fontsize=11, fontweight='bold', pad=10)
    t2 = ax2.table(cellText=summary_rows, colLabels=summary_cols,
                    cellLoc='center', loc='center')
    style_table(t2, len(summary_cols), n_summary)

    fig.tight_layout()
    out_dir = "analysis/map_results"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/combined_lap_stats.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_dir}/combined_lap_stats.png")


plot_combined_lap_stats(lap_comparison, COMBINED_MAPS)


def plot_collision_free_survival(d):
    """Collision-free survival curve over all maps (trials).

    For each agent, every map is one trial.  At each lap index we check whether
    the trial has remained collision-free up to (and including) that lap.
    The y-axis is the fraction of trials still collision-free; the x-axis is
    the lap number.
    """
    import matplotlib.pyplot as plt
    import os

    # Gather per-agent, per-trial collision histories
    # agent -> list of lists, one inner list per trial (map).
    # Each inner list has one bool per lap: True = collision-free that lap.
    agent_trials = {}
    for map_name, agents in d.items():
        for agent_name, data in agents.items():
            if agent_name not in agent_trials:
                agent_trials[agent_name] = []
            cf_flags = []
            for lap in data['laps']:
                cf_flags.append(lap.get('Collisions', 0) == 0)
            if cf_flags:
                agent_trials[agent_name].append(cf_flags)

    if not agent_trials:
        return

    # Determine the maximum lap count across everything
    max_laps = max(
        len(trial) for trials in agent_trials.values() for trial in trials
    )

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (agent_name, trials) in enumerate(sorted(agent_trials.items())):
        n_trials = len(trials)
        # For each lap k (0-indexed), compute fraction of trials that are
        # still collision-free through lap k.
        survival = []
        for k in range(max_laps):
            still_alive = 0
            for trial in trials:
                if k < len(trial):
                    # Collision-free through lap k means all laps 0..k are True
                    if all(trial[:k + 1]):
                        still_alive += 1
                # If trial has fewer laps than k, it either DNF'd or ended;
                # count it as NOT surviving past its last lap.
            survival.append(still_alive / n_trials)

        laps_x = np.arange(1, max_laps + 1)
        color = color_cycle[idx % len(color_cycle)]
        ax.step(laps_x, survival, where='post', linewidth=1.8,
                color=color, label=f"{agent_name} ({n_trials} trials)")

    ax.set_xlabel("Lap", fontsize=11)
    ax.set_ylabel("Fraction Collision-Free", fontsize=11)
    ax.set_title("Collision-Free Survival Curve (All Maps)", fontsize=13,
                 fontweight='bold')
    ax.set_xlim(0.5, max_laps + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, linewidth=0.4, alpha=0.5)

    os.makedirs("analysis/map_results", exist_ok=True)
    fig.tight_layout()
    fig.savefig("analysis/map_results/collision_free_survival.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

plot_collision_free_survival(lap_comparison)


def plot_lap_time_distribution(d):
    """Per-map boxplot of lap-time distributions across agents.

    Each agent gets one box showing the spread of completed lap times.
    DNF laps are excluded from the boxplot but their count is noted in the
    x-tick label so failure modes are still visible.
    """
    import matplotlib.pyplot as plt
    import os

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        agent_names = sorted(agents.keys())
        lap_times_per_agent = []
        labels = []
        for agent_name in agent_names:
            laps = agents[agent_name]['laps']
            times = []
            n_dnf = 0
            for lap in laps:
                t = lap.get('Time')
                if t == 'DNF':
                    n_dnf += 1
                    continue
                try:
                    times.append(float(t))
                except (TypeError, ValueError):
                    pass
            lap_times_per_agent.append(times)
            n_total = len(laps)
            lbl = f"{agent_name}\n({n_total}/{EXPECTED_LAPS} laps)"
            if n_dnf:
                lbl += f"\n[{n_dnf} DNF]"
            labels.append(lbl)

        if not any(lap_times_per_agent):
            continue

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_agents = len(agent_names)

        fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_agents), 5))

        # Only pass non-empty data to boxplot, using explicit positions
        # so agents with all-DNF laps still appear as labelled x-ticks.
        non_empty_data = []
        non_empty_positions = []
        non_empty_colors = []
        for i, times in enumerate(lap_times_per_agent):
            if times:
                non_empty_data.append(times)
                non_empty_positions.append(i + 1)
                non_empty_colors.append(color_cycle[i % len(color_cycle)])

        if non_empty_data:
            bp = ax.boxplot(
                non_empty_data,
                positions=non_empty_positions,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=1.5),
                widths=0.5,
            )

            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(non_empty_colors[j])
                patch.set_alpha(0.65)

        # Scatter individual points
        for i, times in enumerate(lap_times_per_agent):
            if times:
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(times))
                ax.scatter(np.full(len(times), i + 1) + jitter, times,
                           s=18, color=color_cycle[i % len(color_cycle)],
                           edgecolors='black', linewidths=0.4,
                           zorder=5, alpha=0.8)

        ax.set_xlim(0.5, n_agents + 0.5)
        ax.set_xticks(range(1, n_agents + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Lap Time (s)', fontsize=11)
        ax.set_title(f'Lap-Time Distribution — {map_name}  ({len(laps)} trials)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(f"{map_dir}/lap_time_distribution.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved {map_dir}/lap_time_distribution.png")


plot_lap_time_distribution(lap_comparison)

# Now run the slower plot functions
plot_raceline_on_map_image(lap_comparison)
plot_velocity_profiles(lap_comparison)


# ──────────────────────────────────────────────────────────────────────
# Multi-agent interaction metrics (overtake data)
# ──────────────────────────────────────────────────────────────────────

overtake_data = dict()
try:
    with open('analysis/race_data_OVERTAKE.pkl', 'rb') as f:
        overtake_data = pickle.load(f)
    # Rename SupervisedAgent → Ours and drop PurePursuit to match race_data
    if 'SupervisedAgent' in overtake_data:
        overtake_data['TR Agent'] = overtake_data.pop('SupervisedAgent')
    overtake_data.pop('PurePursuit', None)
except FileNotFoundError:
    pass


def plot_interaction_metrics(data):
    """
    Produce a per-map interaction-metrics table (saved as PNG) with columns:
        Method | OSR (%) | Coll./min | d_min (m)

    Metrics
    -------
    OSR  – Overtake Success Rate.  For each opponent, we track the signed
           progress gap (ego − opp) on the centerline with wrap-around.
           An *overtake*  = gap crosses from ≤0 to >+thresh.
           A  *got-passed*= gap crosses from >0  to ≤−thresh.
           OSR = overtakes / max(overtakes + got_passed, 1) × 100.

    Coll./min – number of ego collision onsets per minute of sim time.

    d_min – global minimum Euclidean distance between ego and any opponent.
    """
    if not data:
        return
    import matplotlib.pyplot as plt
    import os

    DT = 0.01           # seconds per obs step
    GAP_THRESH = 0.02    # 2 % of lap – hysteresis to filter side-by-side noise

    centerlines = {}

    for map_name in {m for maps in data.values() for m in maps}:
        if map_name not in centerlines:
            try:
                centerlines[map_name] = _load_centerline(map_name)
            except Exception:
                centerlines[map_name] = None

    # Collect results:  { map_name: { agent: {osr, coll_min, d_min} } }
    results = {}

    def _vectorised_progress(positions_xy, cl_xy, cl_cum_s, lap_length):
        """Return progress array (N,) for positions (N,2) using vectorised nearest-point lookup."""
        # positions_xy: (N, 2), cl_xy: (M, 2)
        dists = np.linalg.norm(
            positions_xy[:, None, :] - cl_xy[None, :, :], axis=2)  # (N, M)
        idxs = np.argmin(dists, axis=1)  # (N,)
        return cl_cum_s[idxs] / lap_length

    OSR_SUBSAMPLE = 10  # check every 10th obs for overtake detection (still 1 ms resolution effectively)

    for agent, maps in data.items():
        for map_name, obss in maps.items():
            if not obss:
                continue
            ego = obss[0].get('ego_idx', 0)
            n_cars = len(obss[0]['poses_x'])
            if n_cars <= 1:
                continue  # single-agent data, skip

            n_obs = len(obss)
            sim_time_min = n_obs * DT / 60.0

            # Build position arrays once: (n_obs, n_cars)
            all_px = np.array([o['poses_x'] for o in obss])  # (n_obs, n_cars)
            all_py = np.array([o['poses_y'] for o in obss])
            all_col = np.array([o['collisions'] for o in obss])  # (n_obs, n_cars)

            # --- Collision onsets (0 → any nonzero = rising edge) ---
            ego_in_col = (all_col[:, ego] != 0).astype(int)
            onsets = np.diff(ego_in_col)
            col_onsets = int(np.sum(onsets == 1))
            coll_min = col_onsets / sim_time_min if sim_time_min > 0 else 0.0

            # --- d_min (vectorised) ---
            ego_xy = np.stack([all_px[:, ego], all_py[:, ego]], axis=1)  # (n_obs, 2)
            d_min = float('inf')
            for j in range(n_cars):
                if j == ego:
                    continue
                opp_xy = np.stack([all_px[:, j], all_py[:, j]], axis=1)
                dists = np.sqrt(((ego_xy - opp_xy)**2).sum(axis=1))
                d_min = min(d_min, float(dists.min()))

            # --- OSR using cumulative progress (lap_count + fractional) ---
            osr = 0.0
            cl_info = centerlines.get(map_name)
            has_lap_counts = 'lap_counts' in obss[0]
            if cl_info is not None:
                cl_xy, cl_cum_s, lap_length = cl_info
                sub_idx = np.arange(0, n_obs, OSR_SUBSAMPLE)
                ego_sub_xy = ego_xy[sub_idx]
                ego_frac = _vectorised_progress(ego_sub_xy, cl_xy, cl_cum_s, lap_length)

                # Build cumulative progress: lap_count + fractional position
                if has_lap_counts:
                    all_laps = np.array([o['lap_counts'] for o in obss])  # (n_obs, n_cars)
                    ego_cum = all_laps[sub_idx, ego].astype(float) + ego_frac
                else:
                    ego_cum = ego_frac

                total_overtakes = 0
                total_got_passed = 0
                for j in range(n_cars):
                    if j == ego:
                        continue
                    opp_sub_xy = np.stack([all_px[sub_idx, j], all_py[sub_idx, j]], axis=1)
                    opp_frac = _vectorised_progress(opp_sub_xy, cl_xy, cl_cum_s, lap_length)
                    if has_lap_counts:
                        opp_cum = all_laps[sub_idx, j].astype(float) + opp_frac
                    else:
                        opp_cum = opp_frac

                    # Signed gap: positive = ego ahead
                    gaps = ego_cum - opp_cum

                    # 3-state machine: AHEAD (+1), BEHIND (-1), NEUTRAL (0)
                    if gaps[0] > GAP_THRESH:
                        state = 1
                    elif gaps[0] < -GAP_THRESH:
                        state = -1
                    else:
                        state = 0

                    overtakes_j = 0
                    got_passed_j = 0
                    for k in range(1, len(gaps)):
                        if gaps[k] > GAP_THRESH:
                            if state == -1:
                                overtakes_j += 1
                            state = 1
                        elif gaps[k] < -GAP_THRESH:
                            if state == 1:
                                got_passed_j += 1
                            state = -1
                    total_overtakes += overtakes_j
                    total_got_passed += got_passed_j
                denom = total_overtakes + total_got_passed
                if denom > 0:
                    osr = total_overtakes / denom * 100.0
                else:
                    # No overtake events: fall back to position-based scoring.
                    # What % of opponents is the ego ahead of at the end?
                    n_ahead = 0
                    n_opp = 0
                    for j in range(n_cars):
                        if j == ego:
                            continue
                        n_opp += 1
                        opp_sub_xy = np.stack([all_px[sub_idx[-1:], j], all_py[sub_idx[-1:], j]], axis=1)
                        opp_frac_end = _vectorised_progress(opp_sub_xy, cl_xy, cl_cum_s, lap_length)[0]
                        if has_lap_counts:
                            opp_cum_end = float(all_laps[sub_idx[-1], j]) + opp_frac_end
                        else:
                            opp_cum_end = opp_frac_end
                        if ego_cum[-1] > opp_cum_end:
                            n_ahead += 1
                    osr = (n_ahead / n_opp * 100.0) if n_opp > 0 else 0.0

            results.setdefault(map_name, {})[agent] = {
                'osr': osr,
                'coll_min': coll_min,
                'd_min': d_min,
            }
            print(f"  {agent}/{map_name}: OSR={osr:.1f}%, "
                  f"Coll/min={coll_min:.2f}, d_min={d_min:.2f}m")

    # --- Render table per map ---
    for map_name, agents in results.items():
        col_labels = ['Method', 'OSR (%)', 'Coll./min', r'$d_{\min}$ (m)']
        rows = []
        for agent in sorted(agents):
            m = agents[agent]
            rows.append([
                agent,
                f"{m['osr']:.1f}",
                f"{m['coll_min']:.2f}",
                f"{m['d_min']:.2f}",
            ])

        n_rows = len(rows)
        fig_h = 0.55 + 0.35 * n_rows
        fig, ax = plt.subplots(figsize=(7, fig_h))
        ax.axis('off')
        ax.set_title(f"Interaction Metrics – {map_name}",
                     fontsize=13, fontweight='bold', pad=12)

        table = ax.table(
            cellText=rows, colLabels=col_labels,
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

        # Alternate row shading
        for i in range(1, n_rows + 1):
            for j in range(len(col_labels)):
                cell = table[i, j]
                cell.set_facecolor('#D9E2F3' if i % 2 == 0 else 'white')

        out_dir = f"analysis/map_results/{map_name}"
        os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/interaction_stats.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_dir}/interaction_stats.png")

    # --- Combined summary table across all maps ---
    if results:
        all_agents = sorted({a for agents in results.values() for a in agents})
        col_labels = ['Method', 'OSR (%)', 'Coll./min', r'$d_{\min}$ (m)']
        rows = []
        for agent in all_agents:
            osrs, cms, dms = [], [], []
            for map_name, agents in results.items():
                if agent in agents:
                    m = agents[agent]
                    osrs.append(m['osr'])
                    cms.append(m['coll_min'])
                    dms.append(m['d_min'])
            rows.append([
                agent,
                f"{np.mean(osrs):.1f}" if osrs else "–",
                f"{np.mean(cms):.2f}" if cms else "–",
                f"{np.mean(dms):.2f}" if dms else "–",
            ])

        n_rows = len(rows)
        fig_h = 0.55 + 0.35 * n_rows
        fig, ax = plt.subplots(figsize=(7, fig_h))
        ax.axis('off')
        ax.set_title("Interaction Metrics – All Maps",
                     fontsize=13, fontweight='bold', pad=12)
        table = ax.table(
            cellText=rows, colLabels=col_labels,
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        for i in range(1, n_rows + 1):
            for j in range(len(col_labels)):
                cell = table[i, j]
                cell.set_facecolor('#D9E2F3' if i % 2 == 0 else 'white')

        os.makedirs("analysis/map_results", exist_ok=True)
        fig.tight_layout()
        fig.savefig("analysis/map_results/interaction_stats_summary.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved analysis/map_results/interaction_stats_summary.png")


plot_interaction_metrics(overtake_data)


def plot_overtake_time_series(data):
    """
    Per-agent dual-axis figure: ego speed and min separation distance d_min(t)
    over time.  Illustrates safe interaction during dense-traffic overtaking.
    """
    if not data:
        return
    import matplotlib.pyplot as plt
    import os

    DT = 0.01  # seconds per obs step

    for agent, maps in data.items():
        print(agent)
        for map_name, obss in maps.items():
            if not obss:
                continue
            ego = obss[0].get('ego_idx', 0)
            n_cars = len(obss[0]['poses_x'])
            if n_cars <= 1:
                continue
            n_obs = len(obss)
            n_opp = n_cars - 1

            # Build arrays
            all_px = np.array([o['poses_x'] for o in obss])
            all_py = np.array([o['poses_y'] for o in obss])
            ego_vx = np.array([o['linear_vels_x'][ego] for o in obss])
            all_col = np.array([o['collisions'] for o in obss])
            ego_col = (all_col[:, ego] != 0)

            # d_min(t): min distance to any opponent at each timestep
            ego_xy = np.stack([all_px[:, ego], all_py[:, ego]], axis=1)
            d_min_t = np.full(n_obs, np.inf)
            for j in range(n_cars):
                if j == ego:
                    continue
                opp_xy = np.stack([all_px[:, j], all_py[:, j]], axis=1)
                d_min_t = np.minimum(d_min_t, np.sqrt(((ego_xy - opp_xy)**2).sum(axis=1)))

            time_s = np.arange(n_obs) * DT

            # ── Figure ──
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = ax1.twinx()

            # Speed
            ax1.plot(time_s, ego_vx, color='#2c7bb6', linewidth=1.0,
                     label='Ego speed', zorder=3)
            ax1.set_ylabel('Ego Speed (m/s)', color='#2c7bb6', fontsize=14)
            ax1.tick_params(axis='y', labelcolor='#2c7bb6', labelsize=12)

            # d_min
            ax2.plot(time_s, d_min_t, color='#d7191c', linewidth=0.8,
                     alpha=0.85, label=r'$d_{\min}(t)$', zorder=2)
            ax2.set_ylabel(r'Min Separation $d_{\min}$ (m)', color='#d7191c', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='#d7191c', labelsize=12)

            # Shade collision intervals
            col_changes = np.diff(ego_col.astype(int))
            starts = np.where(col_changes == 1)[0]
            ends = np.where(col_changes == -1)[0]
            if ego_col[0]:
                starts = np.concatenate([[0], starts])
            if ego_col[-1]:
                ends = np.concatenate([ends, [n_obs - 1]])
            for s, e in zip(starts, ends):
                ax1.axvspan(time_s[s], time_s[min(e, n_obs - 1)],
                            color='red', alpha=0.12, zorder=1)
            # Dummy patch for legend
            if len(starts) > 0:
                ax1.fill_between([], [], color='red', alpha=0.12,
                                 label='Collision')

            ax1.set_xlabel('Time (s)', fontsize=14)
            ax1.set_title(
                f'{agent} — {map_name}  ({n_opp} opponents)',
                fontsize=16, fontweight='bold')
            ax1.set_xlim(0, time_s[-1])
            ax1.tick_params(axis='x', labelsize=12)

            # Merge legends
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=12, loc='lower right',
                       framealpha=0.8)

            ax1.grid(True, linewidth=0.3, alpha=0.5)
            fig.tight_layout()

            out_dir = f'analysis/map_results/{map_name}'
            os.makedirs(out_dir, exist_ok=True)
            safe = agent.replace(' ', '_')
            fig.savefig(f'{out_dir}/overtake_timeseries_{safe}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {out_dir}/overtake_timeseries_{safe}.png')


plot_overtake_time_series(overtake_data)