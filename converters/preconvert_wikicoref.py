import sys

def main():
  input_file = sys.argv[1]
  output_file = sys.argv[2]

  lines = []

  curr_doc_title = None
  with open(input_file, 'r') as f:
    for line in f:
      if line.startswith("#begin"):
        curr_doc_title = "_".join(line.split()[2:])
        lines.append(line)
      elif (
          line.startswith("#end")
          or not line.strip()
          or line.strip() == "null"):
        lines.append(line)
      else:
        fields = line.split()
        assert len(fields) == 5
        fields[0] = curr_doc_title
        lines.append("\t".join(fields) + "\n")

  with open(output_file, 'w') as f:
    f.write("".join(lines))


if __name__ == "__main__":
  main()
