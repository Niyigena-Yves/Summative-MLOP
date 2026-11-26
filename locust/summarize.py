"""Summarise Locust CSV results into a markdown table (locust/results/summary.md)."""
import csv
import glob
import os
import re

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "containers_*_stats.csv"))):
    n = int(re.search(r"containers_(\d+)_stats", path).group(1))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["Name"] == "/predict":
                rows.append((n, r))

head = ("| Containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) | Throughput (req/s) |\n"
        "|---|---|---|---|---|---|---|---|---|")
lines = [head]
for n, r in rows:
    lines.append(f"| {n} | {r['Request Count']} | {r['Failure Count']} | {float(r['Average Response Time']):.0f} | "
                 f"{r['Median Response Time']} | {r['95%']} | {r['99%']} | {float(r['Max Response Time']):.0f} | "
                 f"{float(r['Requests/s']):.2f} |")
out = "\n".join(lines)
with open(os.path.join(HERE, "summary.md"), "w") as f:
    f.write(out + "\n")
print(out)
