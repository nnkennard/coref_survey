import convert_lib
import sys
import os

def read_in_preco(data_home):
  preco_directory = os.path.join(data_home, "original", "PreCo_1.0")

  preco_datasets = {}

  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev]:
    input_filename = os.path.join(preco_directory, split + "." +
        convert_lib.FormatName.jsonl)
    preco_datasets[(convert_lib.DatasetName.preco, split)] = convert_lib.dataset_from_preco(input_filename)

  return preco_datasets

def read_in_gap(data_home):
  gap_directory = os.path.join(data_home, "original", "GAP", "gap-coreference")

  gap_datasets = {}

  for split, split_name in zip(
      [convert_lib.DatasetSplit.dev, convert_lib.DatasetSplit.valid,
        convert_lib.DatasetSplit.test], ["development", "validation", "test"]):
    input_filename = os.path.join(gap_directory, "gap-" + split_name + ".tsv")
    gap_datasets[(convert_lib.DatasetName.gap, split)] = convert_lib.dataset_from_gap(input_filename)
  return gap_datasets

def read_in_wikicoref(data_home):
  test_set = os.path.join(
    data_home, "original", "WikiCoref", "Evaluation", "key-OntoNotesScheme")
  return {(convert_lib.DatasetName.wikicoref,
    convert_lib.DatasetSplit.test): convert_lib.dataset_from_wikicoref(test_set)}


def write_to_format(data_home, dataset, dataset_name, split, format_name):
  print("Writing ", dataset_name, " to format ", format_name)
  output_filename = convert_lib.get_filename(
      data_home, dataset_name, split, format_name)
  if format_name == convert_lib.FormatName.jsonl:
    with open(output_filename, 'w') as f:
      f.write(dataset.DUMP_FUNCTIONS[format_name]())
  else:
    dataset.DUMP_FUNCTIONS[format_name](output_filename)


INPUT_FUNCTIONS = {
    convert_lib.DatasetName.gap: read_in_gap,
    convert_lib.DatasetName.preco: read_in_preco,
    convert_lib.DatasetName.wikicoref: read_in_wikicoref}


def main():
  data_home = sys.argv[1]
  datasets = {}
  for dataset in [
#      convert_lib.DatasetName.gap,
#      convert_lib.DatasetName.wikicoref,
    convert_lib.DatasetName.preco
      ]:
    datasets.update(INPUT_FUNCTIONS[dataset](data_home))

  for (dataset_name, split), dataset in datasets.items():
    write_to_format(data_home, dataset, dataset_name, split,
        convert_lib.FormatName.jsonl)
    write_to_format(data_home, dataset, dataset_name, split,
        convert_lib.FormatName.stanford)


if __name__ == "__main__":
  main()
