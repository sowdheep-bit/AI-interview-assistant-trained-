import re
with open('scripts/report/report.html', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

print("Metrics in report.html:")
for val, lbl in re.findall(r'<div class="metric-val">([^<]+)</div>\s*<div class="metric-lbl">([^<]+)</div>', text):
    print(f"  {lbl.strip()}: {val.strip()}")

print("\nTable Rows in report.html:")
for row in re.findall(r'<tr>(.*?)</tr>', text, re.DOTALL):
    cells = [re.sub(r'<.*?>', '', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)]
    if cells:
        print(f"  {cells}")
