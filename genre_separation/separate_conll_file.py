import sys
import collections

genre_lines = collections.defaultdict(list)

def main():

  input_file = sys.argv[1]

  IN_DOCUMENT = False
  document_genre = None

  with open(input_file, 'r') as f:
    input_lines = f.readlines()

  for i, line in enumerate(input_lines):
    if line.startswith("#begin"):
      genre = line.split()[2].strip('();').split('/')[0]
    genre_lines[genre].append(line)
  
  for genre, lines in genre_lines.items():
    with open(input_file.replace(".conll", "_"+genre+".conll"), 'w') as f:
      f.write("".join(lines))
  


if __name__ == "__main__":
  main()
