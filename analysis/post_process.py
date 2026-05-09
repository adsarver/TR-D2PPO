import os
import numpy as np

try:
    from analysis.result_utils import (
        format_lap_time,
        load_centerline,
        load_race_data,
        new_lap_record,
        project_progress,
        safe_name,
        style_table,
    )
except ImportError:
    from result_utils import (
        format_lap_time,
        load_centerline,
        load_race_data,
        new_lap_record,
        project_progress,
        safe_name,
        style_table,
    )


RACE_DATA_PATH = 'analysis/analysis/race_data_CS677_no_opp.pkl'
OVERTAKE_DATA_PATH = 'analysis/analysis/race_data_CS677_opp.pkl'
COMBINED_MAPS = os.listdir("maps")
EXPECTED_LAPS = 10


def _lap_progress_total(laps, expected_laps=EXPECTED_LAPS):
    progress = sum(np.clip(lap.get('Progress', 1.0), 0.0, 1.0) for lap in laps)
    return min(progress, float(expected_laps))


def _completion_percent(laps, expected_laps=EXPECTED_LAPS):
    if expected_laps <= 0:
        return 0.0
    return _lap_progress_total(laps, expected_laps) / expected_laps * 100.0


def _completed_lap_count(laps):
    n_completed = 0
    for lap in laps:
        try:
            float(lap['Time'])
            n_completed += 1
        except (KeyError, TypeError, ValueError):
            pass
    return n_completed


def create_lap_comparison(race_data):
    lap_comparison = dict()
    centerlines = {}

    for agent, maps in race_data.items():
        for map_name, obss in maps.items():
            if map_name not in lap_comparison:
                lap_comparison[map_name] = dict()

            # Load centerline for progress computation
            if map_name not in centerlines:
                try:
                    centerlines[map_name] = load_centerline(map_name)
                except Exception:
                    centerlines[map_name] = None

            laps = []
            current_lap = new_lap_record()
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

                if lt is not None and prev_lap_time > 0.5 and lt < prev_lap_time - 0.5:
                    current_lap['Time'] = float(prev_lap_time)
                    laps.append(current_lap)
                    current_lap = new_lap_record()

                current_lap['Positions'].append((obs['poses_x'][0], obs['poses_y'][0]))
                current_lap['Velocity'].append(obs['linear_vels_x'][0])

                if lt is not None:
                    prev_lap_time = float(lt)

            # Append last in-progress lap if it has data and wasn't already added
            if current_lap['Positions'] and (not laps or laps[-1] is not current_lap):
                if col_exit:
                    laps.append(current_lap)
                elif prev_lap_time > 0.5:
                    current_lap['Time'] = float(prev_lap_time)
                    laps.append(current_lap)
                else:
                    current_lap['Time'] = 'DNF'
                    current_lap['DNF_reason'] = 'incomplete'
                    laps.append(current_lap)

            SIM_DT = 0.01
            for lap in laps:
                lap['Max Speed'] = float(np.max(lap['Velocity'])) if lap['Velocity'] else 0.0

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

                is_collision_dnf = (lap['Time'] == 'DNF' and lap.get('DNF_reason') == 'collision')
                lap['CSS'] = base_css + (DNF_PENALTY if is_collision_dnf else 0.0)

                if lap['Time'] == 'DNF' and lap['Positions'] and centerlines.get(map_name) is not None:
                    cl_xy, cl_cum_s, lap_length = centerlines[map_name]
                    positions = np.array(lap['Positions'])
                    prog_all = np.array([
                        project_progress(positions[k], cl_xy, cl_cum_s, lap_length)
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
            safe_agent = safe_name(agent_name)
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

            safe_agent = safe_name(agent_name)
            fig.suptitle(f"Velocity Profile — {agent_name} — {map_name}", fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(f"{map_dir}/velocity_profile_{safe_agent}.png", dpi=300)
            plt.close(fig)

# plot_velocity_profiles called after tables (see below)

def plot_lap_stats_table(d):
    import matplotlib.pyplot as plt
    import os

    cr_k = EXPECTED_LAPS

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        agent_names = sorted(agents.keys())

        per_lap_cols = ['Agent', 'Lap', 'Lap Time (s)', 'Max Speed (m/s)',
                        'Collisions', 'CSS', '% Completed', 'Progress @ Fail']
        per_lap_rows = []
        for agent_name in agent_names:
            data = agents[agent_name]
            laps = data['laps']
            for i, lap in enumerate(laps):
                t = lap['Time']
                t_str = format_lap_time(t)
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

            # % completed: sum lap progress / EXPECTED_LAPS, capped at 100%.
            pct = _completion_percent(laps)
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

def plot_combined_lap_stats(d, map_list):
    """Produce a single combined lap-stats figure that aggregates laps across
    all maps in *map_list*.  Per-lap rows show (Map, Lap, …) and the summary
    table shows one row per agent with statistics pooled over every map."""
    import matplotlib.pyplot as plt
    import os

    active_maps = [m for m in map_list if m in d]
    if not active_maps:
        print("  [combined] No matching maps found – skipping.")
        return

    all_agents = sorted({a for m in active_maps for a in d[m]})

    per_lap_cols = ['Agent', 'Map', 'Lap', 'Lap Time (s)', 'Max Speed (m/s)',
                    'Collisions', 'CSS', '% Completed', 'Progress @ Fail']
    per_lap_rows = []
    agent_laps = {a: [] for a in all_agents}

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
                t_str = format_lap_time(t)
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

    cr_k = EXPECTED_LAPS
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

        completed_maps = 0
        evaluated_maps = 0
        for m in active_maps:
            if agent_name not in d[m]:
                continue
            evaluated_maps += 1
            if _completed_lap_count(d[m][agent_name]['laps']) >= cr_k:
                completed_maps += 1
        cr_at_k = f"{completed_maps / evaluated_maps * 100.0:.1f}%" if evaluated_maps else '0.0%'

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

        # Per-map: sum lap progress / EXPECTED_LAPS, then average across maps.
        per_map_pcts = []
        for m in active_maps:
            if agent_name not in d[m]:
                continue
            map_laps = d[m][agent_name]['laps']
            per_map_pcts.append(_completion_percent(map_laps))
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
        survival = []
        for k in range(max_laps):
            still_alive = 0
            for trial in trials:
                if k < len(trial):
                    # Collision-free through lap k means all laps 0..k are True
                    if all(trial[:k + 1]):
                        still_alive += 1
                # Short trials do not survive beyond their last lap.
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


def plot_lap_time_by_lap(d):
    """Per-map line plot of lap time versus lap number for all agents.

    Completed laps are plotted as connected points. DNF laps are shown with
    an x-marker near the top of the axis so failed attempts remain visible
    without being treated as real lap-time values.
    """
    import matplotlib.pyplot as plt
    import os

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for map_name, agents in d.items():
        map_dir = f"analysis/map_results/{map_name}"
        os.makedirs(map_dir, exist_ok=True)

        agent_names = sorted(agents.keys())
        series = {}
        completed_times_all = []
        max_laps = 0

        for agent_name in agent_names:
            laps = agents[agent_name]['laps']
            max_laps = max(max_laps, len(laps))

            completed_x = []
            completed_y = []
            dnf_x = []
            for lap_idx, lap in enumerate(laps, start=1):
                try:
                    lap_time = float(lap['Time'])
                except (KeyError, TypeError, ValueError):
                    dnf_x.append(lap_idx)
                    continue
                completed_x.append(lap_idx)
                completed_y.append(lap_time)
                completed_times_all.append(lap_time)

            series[agent_name] = {
                'completed_x': completed_x,
                'completed_y': completed_y,
                'dnf_x': dnf_x,
                'n_laps': len(laps),
            }

        if max_laps == 0:
            continue

        if completed_times_all:
            y_min = min(completed_times_all)
            y_max = max(completed_times_all)
            y_range = max(y_max - y_min, 1.0)
            dnf_y = y_max + 0.08 * y_range
            ylim_top = y_max + 0.18 * y_range
            ylim_bottom = max(0.0, y_min - 0.10 * y_range)
        else:
            dnf_y = 1.0
            ylim_bottom = 0.0
            ylim_top = 1.2

        fig, ax = plt.subplots(figsize=(max(7, 0.55 * max_laps), 5))

        for idx, agent_name in enumerate(agent_names):
            color = color_cycle[idx % len(color_cycle)]
            data = series[agent_name]
            label = f"{agent_name} ({data['n_laps']}/{EXPECTED_LAPS} laps)"

            if data['completed_x']:
                ax.plot(
                    data['completed_x'],
                    data['completed_y'],
                    marker='o',
                    markersize=4,
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            else:
                ax.plot([], [], marker='o', linewidth=1.5,
                        color=color, label=label)

            if data['dnf_x']:
                ax.scatter(
                    data['dnf_x'],
                    np.full(len(data['dnf_x']), dnf_y),
                    marker='x',
                    s=50,
                    linewidths=1.5,
                    color=color,
                    zorder=5,
                )

        if any(series[a]['dnf_x'] for a in agent_names):
            ax.axhline(dnf_y, color='black', linewidth=0.6,
                       linestyle='--', alpha=0.35)
            ax.text(
                0.01,
                dnf_y,
                'DNF',
                transform=ax.get_yaxis_transform(),
                ha='left',
                va='bottom',
                fontsize=9,
                color='black',
            )

        ax.set_xlim(0.5, max_laps + 0.5)
        ax.set_ylim(ylim_bottom, ylim_top)
        ax.set_xticks(range(1, max_laps + 1))
        ax.set_xlabel('Lap Number', fontsize=11)
        ax.set_ylabel('Lap Time (s)', fontsize=11)
        ax.set_title(f'Lap Time over Lap Number — {map_name}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, linewidth=0.35, alpha=0.5)
        ax.legend(fontsize=9, loc='best')
        fig.tight_layout()
        fig.savefig(f"{map_dir}/lap_time_by_lap.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved {map_dir}/lap_time_by_lap.png")


def load_overtake_data():
    try:
        data = load_race_data(
            OVERTAKE_DATA_PATH,
            agent_renames={'SupervisedAgent': 'TR Agent'},
        )
    except FileNotFoundError:
        print(f"  Overtake data file not found at {OVERTAKE_DATA_PATH} – skipping interaction metrics.")
        return {}
    data.pop('PurePursuit', None)
    return data


def plot_interaction_metrics(data):
    """Save per-map and aggregate overtaking interaction tables."""
    if not data:
        print("  No overtake data found – skipping interaction metrics.")
        return
    import matplotlib.pyplot as plt
    import os

    DT = 0.01           # seconds per obs step
    GAP_THRESH = 0.02    # 2 % of lap – hysteresis to filter side-by-side noise

    centerlines = {}

    for map_name in {m for maps in data.values() for m in maps}:
        if map_name not in centerlines:
            try:
                centerlines[map_name] = load_centerline(map_name)
            except Exception:
                centerlines[map_name] = None

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

            all_px = np.array([o['poses_x'] for o in obss])  # (n_obs, n_cars)
            all_py = np.array([o['poses_y'] for o in obss])
            all_col = np.array([o['collisions'] for o in obss])  # (n_obs, n_cars)

            ego_in_col = (all_col[:, ego] != 0).astype(int)
            onsets = np.diff(ego_in_col)
            col_onsets = int(np.sum(onsets == 1))
            coll_min = col_onsets / sim_time_min if sim_time_min > 0 else 0.0

            ego_xy = np.stack([all_px[:, ego], all_py[:, ego]], axis=1)  # (n_obs, 2)
            d_min = float('inf')
            for j in range(n_cars):
                if j == ego:
                    continue
                opp_xy = np.stack([all_px[:, j], all_py[:, j]], axis=1)
                dists = np.sqrt(((ego_xy - opp_xy)**2).sum(axis=1))
                d_min = min(d_min, float(dists.min()))

            osr = 0.0
            cl_info = centerlines.get(map_name)
            has_lap_counts = 'lap_counts' in obss[0]
            if cl_info is not None:
                cl_xy, cl_cum_s, lap_length = cl_info
                sub_idx = np.arange(0, n_obs, OSR_SUBSAMPLE)
                ego_sub_xy = ego_xy[sub_idx]
                ego_frac = _vectorised_progress(ego_sub_xy, cl_xy, cl_cum_s, lap_length)

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
        style_table(
            table,
            len(col_labels),
            n_rows,
            font_size=10,
            header_color='#4472C4',
            stripe_color='#D9E2F3',
            scale=(1.0, 1.5),
        )

        out_dir = f"analysis/map_results/{map_name}"
        os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/interaction_stats.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_dir}/interaction_stats.png")

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
        style_table(
            table,
            len(col_labels),
            n_rows,
            font_size=10,
            header_color='#4472C4',
            stripe_color='#D9E2F3',
            scale=(1.0, 1.5),
        )

        os.makedirs("analysis/map_results", exist_ok=True)
        fig.tight_layout()
        fig.savefig("analysis/map_results/interaction_stats_summary.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved analysis/map_results/interaction_stats_summary.png")

def plot_overtake_time_series(data):
    """
    Per-agent dual-axis figure: ego speed and min separation distance d_min(t)
    over time.  Illustrates safe interaction during dense-traffic overtaking.
    """
    if not data:
        print("  No overtake data found – skipping interaction metrics.")
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
            safe = safe_name(agent)
            fig.savefig(f'{out_dir}/overtake_timeseries_{safe}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {out_dir}/overtake_timeseries_{safe}.png')


def plot_overtake_snapshots(data, target_agent=None,
                            lookback=120, lookahead=90,
                            zoom_radius=4.0, n_overtakes_max=3):
    """
    Render 4-phase bird's-eye snapshots of overtaking events.

    This follows the previous BC-LSTM paper visualization style. Each detected
    close-pass event is shown as Approaching, Getting Beside, Alongside, and
    Past. For multi-opponent traffic runs, the closest opponent at the event
    center is selected and tracked across the four panels.
    """
    if not data:
        print("  No overtake data found - skipping overtake snapshots.")
        return
    if target_agent is not None and target_agent not in data:
        print(f"  [overtake_snapshots] No data for '{target_agent}' - skipping.")
        return

    import glob
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as patheffects
    import os
    import yaml
    from PIL import Image

    try:
        from scipy.signal import argrelextrema
    except Exception:
        argrelextrema = None

    DT = 0.01
    PROXIMITY_THRESH = 5.0
    MIN_EVENT_SEP = int(4.0 / DT)
    WARMUP_SKIP = 0

    NEAR_THRESH = 0.5
    FAR_THRESH = 1.0
    PHASE_LABELS = ['Approaching', 'Getting Beside', 'Alongside', 'Past']

    def _local_minima(values, order):
        if argrelextrema is not None:
            return argrelextrema(values, np.less_equal, order=order)[0]
        minima = []
        for idx in range(order, len(values) - order):
            window = values[idx - order:idx + order + 1]
            if values[idx] <= np.min(window):
                minima.append(idx)
        return np.array(minima, dtype=int)

    def _load_raceline(map_name):
        try:
            paths = glob.glob(f'maps/{map_name}/*_raceline.csv')
            if not paths:
                return None
            raw = np.genfromtxt(paths[0], delimiter=';', comments='#')
            if raw.ndim != 2 or raw.shape[1] < 3:
                return None
            return raw[:, 1:3]
        except Exception:
            return None

    def _load_map_image(map_name):
        try:
            image_path = f"maps/{map_name}/{map_name}_map.png"
            if not os.path.exists(image_path):
                candidates = glob.glob(f"maps/{map_name}/*_map.png")
                if not candidates:
                    return None
                image_path = candidates[0]
            yaml_path = image_path.replace('_map.png', '_map.yaml')
            if not os.path.exists(yaml_path):
                candidates = glob.glob(f"maps/{map_name}/*_map.yaml")
                if not candidates:
                    return None
                yaml_path = candidates[0]

            image = Image.open(image_path)
            image_arr = np.array(image)
            with open(yaml_path, 'r') as file:
                map_yaml = yaml.safe_load(file)
            origin = map_yaml['origin']
            resolution = map_yaml['resolution']
            img_width, img_height = image.size
            return {
                'image': image_arr,
                'x_min': origin[0],
                'y_min': origin[1],
                'x_max': origin[0] + img_width * resolution,
                'y_max': origin[1] + img_height * resolution,
            }
        except Exception as exc:
            print(f"  [overtake_snapshots] {map_name}: could not load map image ({exc})")
            return None

    agents_to_plot = [target_agent] if target_agent is not None else sorted(data.keys())
    for agent in agents_to_plot:
        for map_name, obss in data.get(agent, {}).items():
            if not obss:
                continue
            ego = obss[0].get('ego_idx', 0)
            n_cars = len(obss[0]['poses_x'])
            if n_cars <= 1:
                continue

            n_obs = len(obss)
            all_px = np.array([o['poses_x'] for o in obss])
            all_py = np.array([o['poses_y'] for o in obss])
            ego_xy = np.stack([all_px[:, ego], all_py[:, ego]], axis=1)

            opponent_indices = [idx for idx in range(n_cars) if idx != ego]
            opponent_distances = []
            for opp_idx in opponent_indices:
                opp_xy = np.stack([all_px[:, opp_idx], all_py[:, opp_idx]], axis=1)
                opponent_distances.append(np.sqrt(((ego_xy - opp_xy)**2).sum(axis=1)))
            opponent_distances = np.stack(opponent_distances, axis=1)
            d_min_t = np.min(opponent_distances, axis=1)
            closest_opp_t = np.argmin(opponent_distances, axis=1)

            kernel = int(0.5 / DT)
            if kernel > 1:
                d_smooth = np.convolve(d_min_t, np.ones(kernel) / kernel, mode='same')
            else:
                d_smooth = d_min_t

            order = max(1, int(1.0 / DT))
            local_min_idx = _local_minima(d_smooth, order)
            close_events = [int(idx) for idx in local_min_idx
                            if d_min_t[idx] < PROXIMITY_THRESH and idx >= WARMUP_SKIP]

            if not close_events:
                print(f"  [overtake_snapshots] {agent}/{map_name}: no close-proximity events found - skipping.")
                continue

            deduped = [close_events[0]]
            for step in close_events[1:]:
                if step - deduped[-1] > MIN_EVENT_SEP:
                    deduped.append(step)
                elif d_min_t[step] < d_min_t[deduped[-1]]:
                    deduped[-1] = step

            deduped.sort(key=lambda idx: d_min_t[idx])
            event_centers = sorted(deduped[:n_overtakes_max])

            event_rows = []
            for center in event_centers:
                opp_col = int(closest_opp_t[center])
                opp_idx = opponent_indices[opp_col]
                distances_to_opp = opponent_distances[:, opp_col]

                window_thresh = max(PROXIMITY_THRESH, FAR_THRESH * 3.0)
                win_start = center
                while win_start > 0 and distances_to_opp[win_start - 1] < window_thresh:
                    win_start -= 1
                win_end = center
                while win_end < n_obs - 1 and distances_to_opp[win_end + 1] < window_thresh:
                    win_end += 1

                min_half = int(0.3 / DT)
                win_start = min(win_start, max(0, center - min_half))
                win_end = max(win_end, min(n_obs - 1, center + min_half))

                total = win_end - win_start
                phase_steps = [
                    win_start,
                    win_start + total // 3,
                    center,
                    min(win_end, n_obs - 1),
                ]
                for phase_idx in range(1, 4):
                    if phase_steps[phase_idx] <= phase_steps[phase_idx - 1]:
                        phase_steps[phase_idx] = min(
                            phase_steps[phase_idx - 1] + max(1, int(0.1 / DT)),
                            n_obs - 1,
                        )
                event_rows.append({
                    'opp_idx': opp_idx,
                    'opp_col': opp_col,
                    'steps': phase_steps,
                })

            map_image = _load_map_image(map_name)
            if map_image is None:
                print(f"  [overtake_snapshots] {agent}/{map_name}: map image unavailable - skipping.")
                continue
            raceline_xy = _load_raceline(map_name)

            out_dir = f"analysis/map_results/{map_name}"
            os.makedirs(out_dir, exist_ok=True)

            n_events = len(event_rows)
            fig, axes = plt.subplots(
                n_events,
                4,
                figsize=(5.5 * 4, 5.0 * n_events),
                squeeze=False,
            )

            for row_idx, event in enumerate(event_rows):
                opp_idx = event['opp_idx']
                opp_col = event['opp_col']
                for col_idx, step in enumerate(event['steps']):
                    ax = axes[row_idx][col_idx]

                    ego_x = all_px[step, ego]
                    ego_y = all_py[step, ego]
                    opp_x = all_px[step, opp_idx]
                    opp_y = all_py[step, opp_idx]
                    cx = opp_x
                    cy = opp_y

                    sep = np.sqrt((ego_x - opp_x)**2 + (ego_y - opp_y)**2)
                    view_radius = max(zoom_radius, sep + 1.0)

                    ax.imshow(
                        map_image['image'],
                        extent=[map_image['x_min'], map_image['x_max'],
                                map_image['y_min'], map_image['y_max']],
                        aspect='equal',
                        cmap='gray',
                        origin='upper',
                        zorder=0,
                    )

                    if raceline_xy is not None:
                        ax.plot(
                            raceline_xy[:, 0], raceline_xy[:, 1],
                            color='#AB47BC', linewidth=1.4, alpha=0.55,
                            linestyle='-', zorder=1,
                            label='Raceline' if row_idx == 0 and col_idx == 0 else None,
                        )

                    bar_len = 1.0
                    bar_x0 = cx - view_radius + 0.15 * view_radius
                    bar_y0 = cy - view_radius + 0.12 * view_radius
                    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0],
                            color='white', linewidth=2.5,
                            solid_capstyle='butt', zorder=15)
                    ax.text(
                        bar_x0 + bar_len / 2,
                        bar_y0 + 0.08 * view_radius,
                        f'{bar_len:.0f} m',
                        color='white', fontsize=12, ha='center', va='bottom',
                        fontweight='bold', zorder=15,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground='black')],
                    )

                    far_circle = mpatches.Circle(
                        (opp_x, opp_y), radius=FAR_THRESH,
                        fill=True, facecolor='#FFF3E0', alpha=0.35,
                        edgecolor='#FF9800', linewidth=2.5,
                        linestyle='--', zorder=8,
                    )
                    ax.add_patch(far_circle)
                    near_circle = mpatches.Circle(
                        (opp_x, opp_y), radius=NEAR_THRESH,
                        fill=True, facecolor='#FFCDD2', alpha=0.45,
                        edgecolor='#D32F2F', linewidth=2.5,
                        linestyle='-', zorder=9,
                    )
                    ax.add_patch(near_circle)

                    t0 = max(0, step - lookback)
                    ax.plot(
                        all_px[t0:step + 1, ego], all_py[t0:step + 1, ego],
                        color='#2196F3', linewidth=1.8,
                        linestyle='--', alpha=0.75, zorder=10,
                        label='Ego past' if row_idx == 0 and col_idx == 0 else None,
                    )

                    t1 = min(n_obs, step + lookahead)
                    future_x = all_px[step:t1, ego]
                    future_y = all_py[step:t1, ego]
                    ax.plot(
                        future_x, future_y, color='#4CAF50', linewidth=2.5,
                        alpha=0.9, zorder=11,
                        label='Ego future' if row_idx == 0 and col_idx == 0 else None,
                    )
                    if len(future_x) > 2:
                        ax.annotate(
                            '', xy=(future_x[-1], future_y[-1]),
                            xytext=(future_x[0], future_y[0]),
                            arrowprops=dict(
                                arrowstyle='-|>', color='#4CAF50', lw=2.6,
                                mutation_scale=18, shrinkA=0, shrinkB=0,
                            ),
                            zorder=12,
                        )

                    for other_idx in opponent_indices:
                        if other_idx == opp_idx:
                            continue
                        ax.plot(
                            all_px[step, other_idx], all_py[step, other_idx],
                            's', color='#9E9E9E', markersize=5,
                            markeredgecolor='black', markeredgewidth=0.6,
                            alpha=0.65, zorder=7,
                            label='Other traffic' if row_idx == 0 and col_idx == 0 and other_idx == opponent_indices[0] else None,
                        )

                    ax.plot(
                        ego_x, ego_y, 'o', color='#FF9800', markersize=8,
                        markeredgecolor='black', markeredgewidth=1.2,
                        zorder=13,
                        label=f'Ego ({agent})' if row_idx == 0 and col_idx == 0 else None,
                    )
                    ax.plot(
                        opp_x, opp_y, 's', color='#E53935', markersize=7,
                        markeredgecolor='black', markeredgewidth=1.0,
                        zorder=12,
                        label='Opponent' if row_idx == 0 and col_idx == 0 else None,
                    )

                    if row_idx == 0 and col_idx == 0:
                        ax.plot([], [], color='#D32F2F', linewidth=2.5,
                                linestyle='-', label=f'Avoidance zone ({NEAR_THRESH}m)')
                        ax.plot([], [], color='#FF9800', linewidth=2.5,
                                linestyle='--', label=f'Transition zone ({FAR_THRESH}m)')

                    ax.set_xlim(cx - view_radius, cx + view_radius)
                    ax.set_ylim(cy - view_radius, cy + view_radius)
                    ax.set_aspect('equal')
                    distance_at_step = opponent_distances[step, opp_col]
                    ax.set_title(
                        f"{PHASE_LABELS[col_idx]}\nt = {step * DT:.2f}s   d = {distance_at_step:.2f}m",
                        fontsize=14,
                        fontweight='bold',
                    )
                    ax.tick_params(labelsize=10)

            axes[0][0].legend(fontsize=10, loc='upper left', framealpha=0.9,
                              handlelength=1.5)

            safe = safe_name(agent)
            fig.suptitle(f"{agent} - Overtake Sequence - {map_name}",
                         fontsize=20, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(f"{out_dir}/overtake_snapshots_{safe}.png",
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved {out_dir}/overtake_snapshots_{safe}.png")


def plot_first_overtake_overlay_snapshots(data, agents=None,
                                          lookback=120, lookahead=90,
                                          zoom_radius=4.0):
    """Overlay each agent's first detected pass on one 4-phase snapshot sheet.

    Each map gets one comparison figure. For every selected ego policy, the
    first close-proximity event that looks like a pass is chosen; if no clean
    behind-to-ahead transition is detected, the earliest close pass is used as
    a fallback so every available agent remains visible.
    """
    if not data:
        print("  No overtake data found - skipping first-overtake overlays.")
        return

    import glob
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as patheffects
    import os
    import yaml
    from PIL import Image

    try:
        from scipy.signal import argrelextrema
    except Exception:
        argrelextrema = None

    DT = 0.01
    PROXIMITY_THRESH = 5.0
    MIN_EVENT_SEP = int(4.0 / DT)
    WARMUP_SKIP = 0
    GAP_THRESH = 0.02

    NEAR_THRESH = 0.5
    FAR_THRESH = 1.0
    PHASE_LABELS = ['Approaching', 'Getting Beside', 'Alongside', 'Past']
    AGENT_COLORS = {
        'BC_LSTM': '#1f77b4',
        'D2PPO': '#d62728',
        'GFPP': '#2ca02c',
        'MPC': '#9467bd',
        'TR Agent': '#ff7f0e',
    }

    def _local_minima(values, order):
        if argrelextrema is not None:
            return argrelextrema(values, np.less_equal, order=order)[0]
        minima = []
        for idx in range(order, len(values) - order):
            window = values[idx - order:idx + order + 1]
            if values[idx] <= np.min(window):
                minima.append(idx)
        return np.array(minima, dtype=int)

    def _load_raceline(map_name):
        try:
            paths = glob.glob(f'maps/{map_name}/*_raceline.csv')
            if not paths:
                return None
            raw = np.genfromtxt(paths[0], delimiter=';', comments='#')
            if raw.ndim != 2 or raw.shape[1] < 3:
                return None
            return raw[:, 1:3]
        except Exception:
            return None

    def _load_map_image(map_name):
        try:
            image_path = f"maps/{map_name}/{map_name}_map.png"
            if not os.path.exists(image_path):
                candidates = glob.glob(f"maps/{map_name}/*_map.png")
                if not candidates:
                    return None
                image_path = candidates[0]
            yaml_path = image_path.replace('_map.png', '_map.yaml')
            if not os.path.exists(yaml_path):
                candidates = glob.glob(f"maps/{map_name}/*_map.yaml")
                if not candidates:
                    return None
                yaml_path = candidates[0]

            image = Image.open(image_path)
            image_arr = np.array(image)
            with open(yaml_path, 'r') as file:
                map_yaml = yaml.safe_load(file)
            origin = map_yaml['origin']
            resolution = map_yaml['resolution']
            img_width, img_height = image.size
            return {
                'image': image_arr,
                'x_min': origin[0],
                'y_min': origin[1],
                'x_max': origin[0] + img_width * resolution,
                'y_max': origin[1] + img_height * resolution,
            }
        except Exception as exc:
            print(f"  [first_overtake_overlay] {map_name}: could not load map image ({exc})")
            return None

    centerline_cache = {}

    def _progress_at(map_name, obss, all_px, all_py, car_idx, step):
        if map_name not in centerline_cache:
            try:
                centerline_cache[map_name] = load_centerline(map_name)
            except Exception:
                centerline_cache[map_name] = None
        centerline = centerline_cache.get(map_name)
        if centerline is None:
            return None
        cl_xy, cl_cum_s, lap_length = centerline
        frac = project_progress((all_px[step, car_idx], all_py[step, car_idx]),
                                cl_xy, cl_cum_s, lap_length)
        if 'lap_counts' in obss[step]:
            return float(obss[step]['lap_counts'][car_idx]) + frac
        return frac

    def _event_window(center, distances_to_opp, n_obs):
        window_thresh = max(PROXIMITY_THRESH, FAR_THRESH * 3.0)
        win_start = center
        while win_start > 0 and distances_to_opp[win_start - 1] < window_thresh:
            win_start -= 1
        win_end = center
        while win_end < n_obs - 1 and distances_to_opp[win_end + 1] < window_thresh:
            win_end += 1

        min_half = int(0.3 / DT)
        win_start = min(win_start, max(0, center - min_half))
        win_end = max(win_end, min(n_obs - 1, center + min_half))

        total = win_end - win_start
        phase_steps = [
            win_start,
            win_start + total // 3,
            center,
            min(win_end, n_obs - 1),
        ]
        for phase_idx in range(1, 4):
            if phase_steps[phase_idx] <= phase_steps[phase_idx - 1]:
                phase_steps[phase_idx] = min(
                    phase_steps[phase_idx - 1] + max(1, int(0.1 / DT)),
                    n_obs - 1,
                )
        return win_start, win_end, phase_steps

    def _cumulative_progress_series(map_name, obss, all_px, all_py, car_idx):
        if map_name not in centerline_cache:
            try:
                centerline_cache[map_name] = load_centerline(map_name)
            except Exception:
                centerline_cache[map_name] = None
        centerline = centerline_cache.get(map_name)
        if centerline is None:
            return None

        cl_xy, cl_cum_s, lap_length = centerline
        progress = np.array([
            project_progress((all_px[step, car_idx], all_py[step, car_idx]),
                             cl_xy, cl_cum_s, lap_length)
            for step in range(len(obss))
        ])
        if 'lap_counts' in obss[0]:
            lap_counts = np.array([o['lap_counts'][car_idx] for o in obss], dtype=float)
            return lap_counts + progress
        return progress

    def _make_record(agent, ego, opp_idx, opp_col, opponent_indices,
                     opponent_distances, all_px, all_py, center, n_obs,
                     is_pass):
        distances_to_opp = opponent_distances[:, opp_col]
        _, _, phase_steps = _event_window(center, distances_to_opp, n_obs)
        return {
            'agent': agent,
            'ego': ego,
            'opp_idx': opp_idx,
            'opp_col': opp_col,
            'opponent_indices': opponent_indices,
            'opponent_distances': opponent_distances,
            'all_px': all_px,
            'all_py': all_py,
            'steps': phase_steps,
            'center': center,
            'n_obs': n_obs,
            'is_pass': is_pass,
        }

    def _first_overtake_event(agent, map_name, obss):
        if not obss:
            return None
        ego = obss[0].get('ego_idx', 0)
        n_cars = len(obss[0]['poses_x'])
        if n_cars <= 1:
            return None

        n_obs = len(obss)
        all_px = np.array([o['poses_x'] for o in obss])
        all_py = np.array([o['poses_y'] for o in obss])
        ego_xy = np.stack([all_px[:, ego], all_py[:, ego]], axis=1)

        opponent_indices = [idx for idx in range(n_cars) if idx != ego]
        opponent_distances = []
        for opp_idx in opponent_indices:
            opp_xy = np.stack([all_px[:, opp_idx], all_py[:, opp_idx]], axis=1)
            opponent_distances.append(np.sqrt(((ego_xy - opp_xy)**2).sum(axis=1)))
        opponent_distances = np.stack(opponent_distances, axis=1)
        d_min_t = np.min(opponent_distances, axis=1)
        closest_opp_t = np.argmin(opponent_distances, axis=1)

        kernel = int(0.5 / DT)
        if kernel > 1:
            d_smooth = np.convolve(d_min_t, np.ones(kernel) / kernel, mode='same')
        else:
            d_smooth = d_min_t

        order = max(1, int(1.0 / DT))
        local_min_idx = _local_minima(d_smooth, order)
        close_events = [int(idx) for idx in local_min_idx
                        if d_min_t[idx] < PROXIMITY_THRESH and idx >= WARMUP_SKIP]

        if not close_events:
            below_thresh = np.where(
                (d_min_t < PROXIMITY_THRESH)
                & (np.arange(n_obs) >= WARMUP_SKIP)
            )[0]
            if below_thresh.size == 0:
                return None
            close_events = [int(below_thresh[0])]

        close_events = sorted(close_events)
        deduped = [close_events[0]]
        for step in close_events[1:]:
            if step - deduped[-1] > MIN_EVENT_SEP:
                deduped.append(step)
            elif d_min_t[step] < d_min_t[deduped[-1]]:
                deduped[-1] = step
        deduped = sorted(deduped)

        ego_progress = _cumulative_progress_series(map_name, obss, all_px, all_py, ego)
        if ego_progress is not None:
            pass_candidates = []
            for opp_col, opp_idx in enumerate(opponent_indices):
                opp_progress = _cumulative_progress_series(map_name, obss, all_px, all_py, opp_idx)
                if opp_progress is None:
                    continue
                gaps = ego_progress - opp_progress
                state = 1 if gaps[0] > GAP_THRESH else (-1 if gaps[0] < -GAP_THRESH else 0)
                for step in range(1, n_obs):
                    if gaps[step] > GAP_THRESH:
                        if state == -1:
                            search_start = max(0, step - int(4.0 / DT))
                            search_end = min(n_obs - 1, step + int(4.0 / DT))
                            distances_to_opp = opponent_distances[:, opp_col]
                            local = distances_to_opp[search_start:search_end + 1]
                            center = search_start + int(np.argmin(local))
                            pass_candidates.append((step, center, opp_col, opp_idx))
                            break
                        state = 1
                    elif gaps[step] < -GAP_THRESH:
                        state = -1

            if pass_candidates:
                _, center, opp_col, opp_idx = min(pass_candidates, key=lambda item: item[0])
                return _make_record(
                    agent, ego, opp_idx, opp_col, opponent_indices,
                    opponent_distances, all_px, all_py, center, n_obs, True,
                )

        fallback = None
        for center in deduped:
            opp_col = int(closest_opp_t[center])
            opp_idx = opponent_indices[opp_col]
            distances_to_opp = opponent_distances[:, opp_col]
            win_start, win_end, _ = _event_window(center, distances_to_opp, n_obs)

            record = _make_record(
                agent, ego, opp_idx, opp_col, opponent_indices,
                opponent_distances, all_px, all_py, center, n_obs, False,
            )
            if fallback is None:
                fallback = record

            ego_start = _progress_at(map_name, obss, all_px, all_py, ego, win_start)
            ego_end = _progress_at(map_name, obss, all_px, all_py, ego, win_end)
            opp_start = _progress_at(map_name, obss, all_px, all_py, opp_idx, win_start)
            opp_end = _progress_at(map_name, obss, all_px, all_py, opp_idx, win_end)
            if None in (ego_start, ego_end, opp_start, opp_end):
                return fallback

            gap_start = ego_start - opp_start
            gap_end = ego_end - opp_end
            if gap_start < -GAP_THRESH and gap_end > GAP_THRESH:
                record['is_pass'] = True
                return record

        return fallback

    agents_to_plot = agents if agents is not None else sorted(data.keys())
    agents_to_plot = [agent for agent in agents_to_plot if agent in data]
    if not agents_to_plot:
        print("  [first_overtake_overlay] No requested agents found - skipping.")
        return

    map_names = sorted({map_name for agent in agents_to_plot
                        for map_name in data.get(agent, {})})
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for map_name in map_names:
        records = []
        for agent in agents_to_plot:
            record = _first_overtake_event(agent, map_name, data.get(agent, {}).get(map_name, []))
            if record is None:
                print(f"  [first_overtake_overlay] {agent}/{map_name}: no close pass found.")
                continue
            records.append(record)

        if not records:
            continue

        map_image = _load_map_image(map_name)
        if map_image is None:
            print(f"  [first_overtake_overlay] {map_name}: map image unavailable - skipping.")
            continue
        raceline_xy = _load_raceline(map_name)

        focus_x = []
        focus_y = []
        for record in records:
            all_px = record['all_px']
            all_py = record['all_py']
            ego = record['ego']
            opp_idx = record['opp_idx']
            for step in record['steps']:
                focus_x.extend([all_px[step, ego], all_px[step, opp_idx]])
                focus_y.extend([all_py[step, ego], all_py[step, opp_idx]])

        cx = float(np.mean(focus_x))
        cy = float(np.mean(focus_y))
        span_x = max(focus_x) - min(focus_x)
        span_y = max(focus_y) - min(focus_y)
        view_radius = max(zoom_radius, span_x / 2 + 1.0, span_y / 2 + 1.0)

        fig, axes = plt.subplots(1, 4, figsize=(5.6 * 4, 5.4), squeeze=False)
        axes = axes[0]

        for col_idx, ax in enumerate(axes):
            ax.imshow(
                map_image['image'],
                extent=[map_image['x_min'], map_image['x_max'],
                        map_image['y_min'], map_image['y_max']],
                aspect='equal', cmap='gray', origin='upper', zorder=0,
            )
            if raceline_xy is not None:
                ax.plot(raceline_xy[:, 0], raceline_xy[:, 1],
                        color='#AB47BC', linewidth=1.4, alpha=0.55,
                        zorder=1, label='Raceline' if col_idx == 0 else None)

            for idx, record in enumerate(records):
                agent = record['agent']
                color = AGENT_COLORS.get(agent, color_cycle[idx % len(color_cycle)])
                all_px = record['all_px']
                all_py = record['all_py']
                ego = record['ego']
                opp_idx = record['opp_idx']
                opp_col = record['opp_col']
                step = record['steps'][col_idx]
                n_obs = record['n_obs']

                ego_x = all_px[step, ego]
                ego_y = all_py[step, ego]
                opp_x = all_px[step, opp_idx]
                opp_y = all_py[step, opp_idx]
                distance_at_step = record['opponent_distances'][step, opp_col]

                far_circle = mpatches.Circle(
                    (opp_x, opp_y), radius=FAR_THRESH,
                    fill=True, facecolor=color, alpha=0.08,
                    edgecolor=color, linewidth=1.5, linestyle='--', zorder=7,
                )
                ax.add_patch(far_circle)
                near_circle = mpatches.Circle(
                    (opp_x, opp_y), radius=NEAR_THRESH,
                    fill=True, facecolor=color, alpha=0.12,
                    edgecolor=color, linewidth=1.8, linestyle='-', zorder=8,
                )
                ax.add_patch(near_circle)

                t0 = max(0, step - lookback)
                ax.plot(all_px[t0:step + 1, ego], all_py[t0:step + 1, ego],
                        color=color, linewidth=1.6, linestyle='--', alpha=0.7,
                        zorder=10)

                t1 = min(n_obs, step + lookahead)
                future_x = all_px[step:t1, ego]
                future_y = all_py[step:t1, ego]
                ax.plot(future_x, future_y, color=color, linewidth=2.2,
                        alpha=0.9, zorder=11,
                        label=agent if col_idx == 0 else None)
                if len(future_x) > 2:
                    ax.annotate(
                        '', xy=(future_x[-1], future_y[-1]),
                        xytext=(future_x[0], future_y[0]),
                        arrowprops=dict(
                            arrowstyle='-|>', color=color, lw=2.1,
                            mutation_scale=16, shrinkA=0, shrinkB=0,
                        ),
                        zorder=12,
                    )

                ax.plot(ego_x, ego_y, 'o', color=color, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.9, zorder=13)
                ax.plot(opp_x, opp_y, 's', color=color, markersize=6,
                        markeredgecolor='white', markeredgewidth=0.9, zorder=12)
                if col_idx in (1, 2):
                    ax.text(
                        opp_x, opp_y, f'{distance_at_step:.2f}m',
                        color='white', fontsize=6, ha='left', va='top', zorder=14,
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')],
                    )

            bar_len = 1.0
            bar_x0 = cx - view_radius + 0.15 * view_radius
            bar_y0 = cy - view_radius + 0.12 * view_radius
            ax.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0],
                    color='white', linewidth=2.5, solid_capstyle='butt', zorder=15)
            ax.text(
                bar_x0 + bar_len / 2,
                bar_y0 + 0.08 * view_radius,
                f'{bar_len:.0f} m',
                color='white', fontsize=11, ha='center', va='bottom',
                fontweight='bold', zorder=15,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')],
            )

            ax.set_xlim(cx - view_radius, cx + view_radius)
            ax.set_ylim(cy - view_radius, cy + view_radius)
            ax.set_aspect('equal')
            ax.set_title(PHASE_LABELS[col_idx], fontsize=14, fontweight='bold')
            ax.tick_params(labelsize=9)

        axes[0].plot([], [], 'o', color='white', markeredgecolor='black',
                     label='Ego marker')
        axes[0].plot([], [], 's', color='white', markeredgecolor='black',
                     label='Overtaken traffic')
        axes[0].plot([], [], color='black', linewidth=1.6, linestyle='--',
                     label='Past path')
        axes[0].plot([], [], color='black', linewidth=2.2,
                     label='Future path')
        axes[0].legend(fontsize=9, loc='upper left', framealpha=0.9,
                       handlelength=1.5)

        n_pass = sum(1 for record in records if record['is_pass'])
        fig.suptitle(
            f"First Overtake Overlay - {map_name} ({n_pass}/{len(records)} pass transitions detected)",
            fontsize=18, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        out_dir = f"analysis/map_results/{map_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/overtake_snapshots_first_overtake_overlay.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    race_data = load_race_data(RACE_DATA_PATH)
    lap_comparison = create_lap_comparison(race_data)
    plot_lap_stats_table(lap_comparison)
    plot_combined_lap_stats(lap_comparison, COMBINED_MAPS)
    plot_collision_free_survival(lap_comparison)
    plot_lap_time_distribution(lap_comparison)
    plot_lap_time_by_lap(lap_comparison)
    plot_raceline_on_map_image(lap_comparison)
    plot_velocity_profiles(lap_comparison)

    overtake_data = load_overtake_data()
    plot_interaction_metrics(overtake_data)
    plot_overtake_time_series(overtake_data)
    plot_overtake_snapshots(overtake_data)
    plot_first_overtake_overlay_snapshots(overtake_data)


if __name__ == "__main__":
    main()