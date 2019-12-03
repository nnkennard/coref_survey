import collections
import json
import sys
import random
import os

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

CATEGORIES = [TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]

TOKEN_WINDOW = 50

class Model(object):
  SPANBERT = "spanbert"
  BERS = "bers"

class Boundaries(object):
  GOLD = "gold"
  PREDICTED = "predicted"

class Split(object):
  TRAIN = "train"
  TEST = "test"

def split_docs(doc_keys):
  doc_key_list = list(doc_keys)
  random.shuffle(doc_key_list)
  boundary = int(0.8 * len(doc_key_list))
  return doc_key_list[:boundary], doc_key_list[boundary:]

def main():
  random.seed(43)
  result_file, dataset = sys.argv[1:]
  with open(result_file, 'r') as f:
    doc_key_list = [json.loads(line.strip())["doc_key"] for line in f.readlines()]
  doc_keys = {}
  doc_keys[Split.TRAIN], doc_keys[Split.TEST] = split_docs(doc_key_list)

  for split in [Split.TRAIN, Split.TEST]:
    filename = "".join([dataset, "_dt_", split, ".keys"])
    with open(filename, 'w') as f:
      f.write("\n".join(doc_keys[split]))


if __name__ == "__main__":
  main()
