from pathlib import Path
import pandas as pd
import json

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGE_FILES = {
    "No merge":     STATS_DIR / "stats_use_cell_by_hour_by_day.csv",
    "Simple merge": STATS_DIR / "stats_use_cell_by_hour_simple_merge_by_day.csv",
    "2G/3G merge":  STATS_DIR / "stats_use_cell_by_hour_2g3g_merge_by_day.csv",
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
  </div>

  <div class="chart-container">
    <canvas id="bar-chart"></canvas>
  </div>

  <script>
    const DATA   = {data_json};
    const LABELS = {labels_json};

    const daySelect     = document.getElementById("day-select");
    const stationSelect = document.getElementById("station-select");
    const ctx           = document.getElementById("bar-chart").getContext("2d");

    Object.keys(DATA).forEach(day => {{
      const opt = document.createElement("option");
      opt.value = day;
      opt.textContent = day;
      daySelect.appendChild(opt);
    }});

    function populateStations(day) {{
      stationSelect.innerHTML = "";
      Object.keys(DATA[day]).forEach(station => {{
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
      const day     = daySelect.value;
      const station = stationSelect.value;
      chart.data.datasets[0].data = DATA[day][station];
      chart.options.plugins.title.text = `Day: ${{day}} — Station: ${{station}}`;
      chart.update();
    }}

    daySelect.addEventListener("change", () => {{
      populateStations(daySelect.value);
      updateChart();
    }});
    stationSelect.addEventListener("change", updateChart);

    if (Object.keys(DATA).length > 0) {{
      const firstDay = Object.keys(DATA)[0];
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


for merge_name, csv_path in MERGE_FILES.items():
    if not csv_path.exists():
        print(f"File not found, skipping: {csv_path}")
        continue

    df = pd.read_csv(csv_path, sep=";", index_col=[0, 1])
    df.index.names = ["day", "cellid"]
    hour_labels = df.columns.tolist()

    data = df_to_json_data(df)
    title = f"Cell use by hour — {merge_name}"

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=json.dumps(data, ensure_ascii=False),
        labels_json=json.dumps(hour_labels, ensure_ascii=False),
    )

    out_path = OUTPUT_DIR / f"{csv_path.stem}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Generated: {out_path}")
