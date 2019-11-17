import sys


lines = []
curr_cluster_number = 13
for line in sys.stdin:
  fields = line.strip().split("\t")
  int_fields = [int(z) for z in fields[1:5]]
  lines.append(
  [fields[0]] + int_fields + fields[5:]
  )

for line in sorted(lines):
  if line[2] != curr_cluster_number:
    curr_cluster_number = line[2]
    sys.stdout.write("\n")
  sys.stdout.write("\t".join(str(i) for i in line) + "\n")
  
