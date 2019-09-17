import convert_lib

import sys

def main():
  filename = "/iesl/canvas/nnayak/coref_datasets/original/WikiCoref/Evaluation/key-OntoNotesScheme"

  #k = convert_lib.dataset_from_wikicoref(filename)
  k = convert_lib.dataset_from_gap(
      "/iesl/canvas/nnayak/coref_datasets/original/GAP/gap-coreference/gap-development.tsv")
  print(k)
  print(k.dump_to_jsonl())


if __name__ == "__main__":
  main()
