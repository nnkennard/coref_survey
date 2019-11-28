import sys
import collections


span_const_labels = collections.defaultdict(list)

for line in sys.stdin:
  if not line.strip():
    continue
  fields = line.strip().split("\t")
  span_const_labels[tuple(fields[:3])].append(fields[7])
  

for k, v in span_const_labels.items():
  #print(k)
  print("\t".join(sorted(collections.Counter(v).keys())))
