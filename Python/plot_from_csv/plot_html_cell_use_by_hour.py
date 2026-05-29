from pathlib import Path
import pandas as pd
import json

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: (users_csv, connections_csv)
MERGE_FILES = {
    "No merge": (
        STATS_DIR / "stats_use_cell_by_hour_by_day.csv",
        STATS_DIR / "stats_connections_cell_by_hour_by_day.csv",
    ),
    "Simple merge": (
        STATS_DIR / "stats_use_cell_by_hour_simple_merge_by_day.csv",
        STATS_DIR / "stats_connections_cell_by_hour_simple_merge_by_day.csv",
    ),
    "2G/3G merge": (
        STATS_DIR / "stats_use_cell_by_hour_2g3g_merge_by_day.csv",
        STATS_DIR / "stats_connections_cell_by_hour_2g3g_merge_by_day.csv",
    ),
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 960px;
      margin: 40px auto;
      padding: 0 20px;
      background: #f5f5f5;
    }}
    h1 {{ color: #333; }}
    .controls {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }}
    label {{ font-size: 1rem; }}
    select {{
      font-size: 1rem;
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid #aaa;
    }}
    .chart-container {{
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      margin-top: 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="controls">
    <label for="day-select">Day :</label>
    <select id="day-select"></select>

    <label for="station-select">Station :</label>
    <select id="station-select"></select>

    <label for="count-select">Count type :</label>
    <select id="count-select">
      <option value="users">Number of users</option>
      <option value="connections">Number of connections</option>
    </select>
  </div>

  <div class="chart-container">
    <canvas id="bar-chart"></canvas>
  </div>

  <script>
    const DATA   = {data_json};
    const LABELS = {labels_json};

    const COUNT_LABELS = {{
      users:       "Number of users",
      connections: "Number of connections"
    }};

    const daySelect     = document.getElementById("day-select");
    const stationSelect = document.getElementById("station-select");
    const countSelect   = document.getElementById("count-select");
    const ctx           = document.getElementById("bar-chart").getContext("2d");

    Object.keys(DATA.users).forEach(day => {{
      const opt = document.createElement("option");
      opt.value = day;
      opt.textContent = day;
      daySelect.appendChild(opt);
    }});

    function populateStations(day) {{
      stationSelect.innerHTML = "";
      const countType = countSelect.value;
      Object.keys(DATA[countType][day]).forEach(station => {{
        const opt = document.createElement("option");
        opt.value = station;
        opt.textContent = station;
        stationSelect.appendChild(opt);
      }});
    }}

    const chart = new Chart(ctx, {{
      type: "bar",
      data: {{
        labels: LABELS,
        datasets: [{{
          label: "Number of users",
          data: [],
          backgroundColor: "rgba(54, 162, 235, 0.7)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          title: {{
            display: true,
            text: "",
            font: {{ size: 15 }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: "Hour of the day" }} }},
          y: {{ title: {{ display: true, text: "Number of users" }}, beginAtZero: true }}
        }}
      }}
    }});

    function updateChart() {{
      const day       = daySelect.value;
      const station   = stationSelect.value;
      const countType = countSelect.value;
      const label     = COUNT_LABELS[countType];

      chart.data.datasets[0].data  = DATA[countType][day][station];
      chart.data.datasets[0].label = label;
      chart.options.plugins.title.text   = `Day: ${{day}} — Station: ${{station}} — ${{label}}`;
      chart.options.scales.y.title.text  = label;
      chart.update();
    }}

    daySelect.addEventListener("change", () => {{
      populateStations(daySelect.value);
      updateChart();
    }});
    stationSelect.addEventListener("change", updateChart);
    countSelect.addEventListener("change", () => {{
      populateStations(daySelect.value);
      updateChart();
    }});

    if (Object.keys(DATA.users).length > 0) {{
      const firstDay = Object.keys(DATA.users)[0];
      populateStations(firstDay);
      updateChart();
    }}
  </script>
</body>
</html>
"""


def df_to_json_data(df: pd.DataFrame) -> dict:
    """Convert MultiIndex DataFrame (day, cellid) to {day: {station: [24 values]}} dict."""
    result = {}
    for day, group in df.groupby(level="day"):
        group = group.droplevel("day")
        result[str(day)] = {str(idx): row.tolist() for idx, row in group.iterrows()}
    return result


for merge_name, (users_csv, connections_csv) in MERGE_FILES.items():
    missing = [p for p in (users_csv, connections_csv) if not p.exists()]
    if missing:
        for p in missing:
            print(f"File not found, skipping: {p}")
        continue

    df_users = pd.read_csv(users_csv, sep=";", index_col=[0, 1])
    df_users.index.names = ["day", "cellid"]

    df_connections = pd.read_csv(connections_csv, sep=";", index_col=[0, 1])
    df_connections.index.names = ["day", "cellid"]

    hour_labels = df_users.columns.tolist()

    data = {
        "users":       df_to_json_data(df_users),
        "connections": df_to_json_data(df_connections),
    }
    title = f"Cell use by hour — {merge_name}"

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=json.dumps(data, ensure_ascii=False),
        labels_json=json.dumps(hour_labels, ensure_ascii=False),
    )

    out_path = OUTPUT_DIR / f"{users_csv.stem}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Generated: {out_path}")
