import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tqdm
import csv
import re
import csv
from utils import get_day

"""
The aim of this srcipt is to have the frequency of use of each cell by hour, 
in order to have a better understanding of the presence of users in the study area during the day, 
and to be able to classify them based on their presence at base stations during the day.
"""

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR / f"Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / f"results/intermediate_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [file for file in INPUT_DIR.glob("*.csv")]

INPUT_CELLS = MAIN_DIR / "Database/cells/cd_142_cells.csv"


def presence_in_cells(cells,stamps,dic_cells):
    """
    This function will create a dictionary with the frequency of use of each cell by hour

    cells : a list of cell ids where the user was present during the day
    stamps : a list of timestamps corresponding to the times when the user was connected at the base stations
    dic_cells : a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values

    return : a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values
    """
    if len(cells) == 0 or len(stamps) == 0:
        return dic_cells

    previous_cell = cells[0]
    previous_stamp = stamps[0]

    passed_hours = {i: False for i in range(24)} # we create a dictionary to keep track of the hours that have already been counted for the current cell

    # morning and evening cases
    firtst_stamp = stamps[0]
    last_stamp = stamps[-1]
    if firtst_stamp // 3600 < 4: # if the first timestamp is before 4am, we consider that the user is present on the first cell until 4am
        for hour in range(0, firtst_stamp // 3600):
            dic_cells[cells[0]][hour] += 1
            passed_hours[hour] = True
    if last_stamp // 3600 >= 20: # if the last timestamp is after 8pm, we consider that the user is present on the last cell until 8pm
        for hour in range(last_stamp // 3600 + 1, 24):
            dic_cells[cells[-1]][hour] += 1
            passed_hours[hour] = True

    for cell, stamp in zip(cells, stamps):
        if cell != previous_cell:
            if stamp - previous_stamp < 4*3600: # if the user is connected to a different cell within 4 hours, we consider that he is still present in the previous cell
                hour_start = previous_stamp // 3600 # we convert the timestamp to hours
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end + 1):
                    if not passed_hours[hour]: # if the hour has not been counted yet, we increment the frequency of use of the previous cell at the corresponding hour
                        dic_cells[previous_cell][hour] += 1 # we increment the frequency of use of the previous cell at the corresponding hour
                        passed_hours[hour] = True # we mark the hour as counted
                previous_cell = cell
                previous_stamp = stamp
            else: # if the user is connected to a different cell after more than 4 hours, we consider that he is present on the previous cell until the end of the hour
                hour= previous_stamp // 3600
                if not passed_hours[hour]: # if the hour has not been counted yet, we increment the frequency of use of the previous cell at the corresponding hour
                    dic_cells[previous_cell][hour] += 1
                    passed_hours[hour] = True
                previous_cell = cell
                previous_stamp = stamp
        else: # if the user is connected to the same cell, we consider that he is present on this cell until the end of the hour
            if stamp - previous_stamp < 4*3600: # if the user is connected to the same cell within 4 hours, we consider that he is still present in this cell
                hour_start = previous_stamp // 3600
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end):
                    if not passed_hours[hour]:
                        dic_cells[cell][hour] += 1 # we increment the frequency of use of the cell at the corresponding hour
                        passed_hours[hour] = True
                previous_stamp = stamp
            else: # if the user is connected to the same cell after more than 4 hours, we consider that he is present on this cell until the end of the hour
                hour = previous_stamp // 3600
                if not passed_hours[hour]:
                    dic_cells[cell][hour] += 1
                    passed_hours[hour] = True
                previous_stamp = stamp
    
    return dic_cells

def get_cell_code(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)

def get_cell_code2(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)[1:]

MERGE = {
    "simple" : get_cell_code,
    "2g3g" : get_cell_code2
}



cells_df = pd.read_csv(INPUT_CELLS, sep=";")
cell_ids = cells_df["cellid"].tolist()
cell_ids_simple = cells_df["cellid"].apply(get_cell_code).unique().tolist()
cell_ids_2g3g = cells_df["cellid"].apply(get_cell_code2).unique().tolist()

hour_cols = [f"{hour}h-{hour+1}h" for hour in range(24)]

all_dfs = []
all_dfs_simple = []
all_dfs_2g3g = []

for file in tqdm.tqdm(files):
    day = get_day(file)

    dict_cells = {cell: [0]*24 for cell in cell_ids}
    dict_cells_simple_merged = {cell: [0]*24 for cell in cell_ids_simple}
    dict_cells_2g3g_merged = {cell: [0]*24 for cell in cell_ids_2g3g}

    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            user_cells = [c for c in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            dict_cells = presence_in_cells(user_cells, user_stamps, dict_cells)
            dict_cells_simple_merged = presence_in_cells([MERGE["simple"](c) for c in user_cells], user_stamps, dict_cells_simple_merged)
            dict_cells_2g3g_merged = presence_in_cells([MERGE["2g3g"](c) for c in user_cells], user_stamps, dict_cells_2g3g_merged)

    df_day = pd.DataFrame.from_dict(dict_cells, orient='index', columns=hour_cols)
    df_day.index.name = "cellid"
    df_day.insert(0, "day", day)
    all_dfs.append(df_day)

    df_day_simple = pd.DataFrame.from_dict(dict_cells_simple_merged, orient='index', columns=hour_cols)
    df_day_simple.index.name = "cellid"
    df_day_simple.insert(0, "day", day)
    all_dfs_simple.append(df_day_simple)

    df_day_2g3g = pd.DataFrame.from_dict(dict_cells_2g3g_merged, orient='index', columns=hour_cols)
    df_day_2g3g.index.name = "cellid"
    df_day_2g3g.insert(0, "day", day)
    all_dfs_2g3g.append(df_day_2g3g)

df = pd.concat(all_dfs).reset_index().set_index(["day", "cellid"])
df_simple_merged = pd.concat(all_dfs_simple).reset_index().set_index(["day", "cellid"])
df_2g3g_merged = pd.concat(all_dfs_2g3g).reset_index().set_index(["day", "cellid"])

df.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour_by_day.csv", sep=";", header=True, index=True)
df_simple_merged.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour_simple_merge_by_day.csv", sep=";", header=True, index=True)
df_2g3g_merged.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour_2g3g_merge_by_day.csv", sep=";", header=True, index=True)

