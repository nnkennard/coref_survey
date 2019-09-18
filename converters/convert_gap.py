import sys
import os

import convert_lib

def main():
  data_home = sys.argv[1]
  gap_directory = os.path.join(data_home, "original", "GAP", "gap-coreference")

  for split, split_name in zip(
      [convert_lib.DatasetSplit.dev, convert_lib.DatasetSplit.valid,
        convert_lib.DatasetSplit.test], ["development", "validation", "test"]):
    input_filename = os.path.join(gap_directory, "gap-" + split_name + ".tsv")
    print(input_filename)
    output_filename = convert_lib.get_filename(
        data_home, convert_lib.DatasetName.gap,
        split_name, convert_lib.FormatName.jsonl)
    with open(output_filename, 'w') as f:
      dataset = convert_lib.dataset_from_gap(input_filename)
      print(dataset.dump_to_jsonl())
      f.write(dataset.dump_to_jsonl())


if __name__ == "__main__":
  main()
