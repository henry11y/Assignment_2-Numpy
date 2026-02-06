
import numpy as np
import csv

CSV_FILE = "basketball.csv"


# ---------- helpers ----------
def safe_divide(a, b):
    """Return a/b; if b is 0 return NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b != 0, a / b, np.nan)


def to_float(col):
    """Convert string column to float; non-numbers become NaN."""
    out = np.empty(col.shape[0], dtype=float)
    for i, x in enumerate(col):
        s = "" if x is None else str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            out[i] = np.nan
        else:
            try:
                out[i] = float(s)
            except ValueError:
                out[i] = np.nan
    return out


def load_csv_loose(filename):
    """
    Reads CSV safely even if some rows have extra/missing columns.
    Returns: headers(list[str]), data(np.ndarray of strings shape [nrows, ncols])
    """
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        ncols = len(headers)
        rows = []
        for r in reader:
            # pad short rows
            if len(r) < ncols:
                r = r + [""] * (ncols - len(r))
            # truncate long rows
            elif len(r) > ncols:
                r = r[:ncols]
            rows.append(r)

    data = np.array(rows, dtype=str)
    headers = [h.strip() for h in headers]
    return headers, data


def col_by_name(headers, data, name):
    """Return a column by header name (case-insensitive)."""
    name_low = name.strip().lower()
    idx = None
    for i, h in enumerate(headers):
        if h.lower() == name_low:
            idx = i
            break
    if idx is None:
        raise KeyError(f"Missing column '{name}'. Found headers like: {headers[:10]} ...")
    return data[:, idx]


def group_sum(values, inv, n_groups):
    """Sum values per group, treating NaN as 0."""
    v = np.nan_to_num(values, nan=0.0)
    return np.bincount(inv, weights=v, minlength=n_groups)


# ---------- main ----------
def main():
    headers, data = load_csv_loose(CSV_FILE)

    # REQUIRED identity columns (edit names here only if your headers differ)
    player = col_by_name(headers, data, "player")
    season = col_by_name(headers, data, "season")

    # Numeric columns (common basketball stat headers)
    # If your file uses different names, change the strings on the right side.
    gp  = to_float(col_by_name(headers, data, "GP"))
    mins = to_float(col_by_name(headers, data, "MIN"))

    fgm = to_float(col_by_name(headers, data, "FGM"))
    fga = to_float(col_by_name(headers, data, "FGA"))

    tpm = to_float(col_by_name(headers, data, "3PM"))
    tpa = to_float(col_by_name(headers, data, "3PA"))

    ftm = to_float(col_by_name(headers, data, "FTM"))
    fta = to_float(col_by_name(headers, data, "FTA"))

    pts = to_float(col_by_name(headers, data, "PTS"))

    # ---- group by (player, season) ----
    key = np.char.add(np.char.add(player, "|"), season)
    uniq_keys, inv = np.unique(key, return_inverse=True)
    n = uniq_keys.size

    # split keys back into player/season
    out_player = np.array([k.split("|", 1)[0] for k in uniq_keys], dtype=str)
    out_season = np.array([k.split("|", 1)[1] for k in uniq_keys], dtype=str)

    # aggregate sums
    sum_gp   = group_sum(gp, inv, n)
    sum_min  = group_sum(mins, inv, n)

    sum_fgm  = group_sum(fgm, inv, n)
    sum_fga  = group_sum(fga, inv, n)

    sum_3pm  = group_sum(tpm, inv, n)
    sum_3pa  = group_sum(tpa, inv, n)

    sum_ftm  = group_sum(ftm, inv, n)
    sum_fta  = group_sum(fta, inv, n)

    sum_pts  = group_sum(pts, inv, n)

    # ---- required calculations ----
    fg_pct = safe_divide(sum_fgm, sum_fga)
    tp_pct = safe_divide(sum_3pm, sum_3pa)
    ft_pct = safe_divide(sum_ftm, sum_fta)

    pts_per_min = safe_divide(sum_pts, sum_min)

    # overall shooting accuracy (made/attempted across FG + 3P + FT)
    overall_made = sum_fgm + sum_3pm + sum_ftm
    overall_att  = sum_fga + sum_3pa + sum_fta
    overall_acc = safe_divide(overall_made, overall_att)

    # ---- build output table ----
    # format % as decimals; you can multiply by 100 if your teacher wants percent numbers
    out = np.column_stack([
        out_player,
        out_season,
        sum_gp.astype(int).astype(str),
        np.round(sum_min, 2).astype(str),
        np.round(fg_pct, 4).astype(str),
        np.round(tp_pct, 4).astype(str),
        np.round(ft_pct, 4).astype(str),
        np.round(pts_per_min, 4).astype(str),
        np.round(overall_acc, 4).astype(str),
    ])

    out_headers = [
        "player", "season", "GP_sum", "MIN_sum",
        "FG_accuracy", "3P_accuracy", "FT_accuracy",
        "points_per_min", "overall_shooting_accuracy"
    ]

    # save results
    out_file = "player_season_results.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_headers)
        w.writerows(out)

    print("Saved:", out_file)
    print("Rows:", out.shape[0])
    print("First 5 rows:")
    for r in out[:5]:
        print(r)


if __name__ == "__main__":
    main()