# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

import collections

span_map = collections.Counter()
pos_map = collections.Counter()
other_count = None

with open("conll_true.tsv", 'r') as f:
  for line in f:
    fields = line.strip().split()
    if int(fields[0]) > 700:
      continue
    if len(fields) == 3:
      count, type, label = line.strip().split()
      if type == "Span":
        span_map[label] = int(count)
      elif type == "POS":
        pos_map[label] = int(count)
    else:
      assert len(fields) == 2
      count, type = fields
      assert type == "Other"
      other_count = int(count)


models = ["CoNLL gold"]

for model in models:
  raw_data = collections.defaultdict(list)
  for span_label, count in span_map.most_common():
    raw_data["Span" + span_label].append(count)
  for span_label, count in pos_map.most_common():
    raw_data["POS" + span_label].append(count)
  raw_data["Other"] = [other_count]


total = sum(span_map.values()) + sum(pos_map.values()) + other_count
print(total)
 
df = pd.DataFrame(raw_data)
 
# From raw value to percentage

keys = list(collections.Counter(raw_data).most_common())
print(keys)

barWidth = 0.85
plotted_so_far = 0

for i, (key, count) in enumerate(keys):
  if key.startswith("Span"):
    plt.bar(0, count, bottom=plotted_so_far, color='#b5ffb9', edgecolor='white', width=barWidth, label=key)
  else:
    plt.bar(0, count, bottom=plotted_so_far, color='#3498db', edgecolor='white', width=barWidth, label=key)

  plt.text(x = 0 , y = plotted_so_far, s = key, size = 6)

  plotted_so_far += count[0]

# Custom x axis
plt.xticks([0], ["CoNLL gold"])
plt.xlabel("group")
 
# Show graphic
plt.show()
