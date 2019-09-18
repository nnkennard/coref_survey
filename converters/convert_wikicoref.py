import sys
import os

import convert_lib

def main():
  data_home = sys.argv[1]
  test_set = os.path.join(
    data_home, "original", "WikiCoref", "Evaluation", "key-OntoNotesScheme")
  output_filename = convert_lib.get_filename(
      data_home, convert_lib.DatasetName.wikicoref,
      convert_lib.DatasetSplit.test, convert_lib.FormatName.jsonl)
  with open(output_filename, 'w') as f:
    dataset = convert_lib.dataset_from_wikicoref(test_set)
    f.write(dataset.dump_to_jsonl())


if __name__ == "__main__":
  main()
