import collections
import json
import sys

def get_bronze_mentions(clusters, predicted_mentions):
  gold_mentions = set(tuple(i) for i in sum(clusters, []))
  print(gold_mentions)
  high_precision = sorted(list(
    predicted_mentions.intersection(gold_mentions)))
  high_recall = sorted(list(
    predicted_mentions.union(gold_mentions)))
  return high_precision, high_recall


def main():
  jsonl_file, predictions_file = sys.argv[1:]

  examples = collections.OrderedDict()
  with open(jsonl_file, 'r') as f:
    for line in f:
      example = json.loads(line.strip())
      examples[example["doc_key"]] = example

  predictions = {}
  with open(predictions_file, 'r') as f:
    for line in f:
      obj = json.loads(line.strip())
      for k, v in obj.items():
        predictions[k] = [(span["start"], span["end"]) for span in v]
  
  high_p_filename = jsonl_file.replace(".jsonl", "_highp.jsonl")
  high_r_filename = jsonl_file.replace(".jsonl", "_highr.jsonl")

  with open(high_p_filename, 'w') as fp:
    with open(high_r_filename, 'w') as fr:
      for doc_key, example in examples.items():
        doc_predictions = set(tuple(i) for i in predictions[doc_key])
        high_precision_mentions, high_recall_mentions = get_bronze_mentions(
          example["clusters"], doc_predictions)
        example["clusters"] = [high_precision_mentions]
        fp.write(json.dumps(example) + "\n")
        example["clusters"] = [high_recall_mentions]
        fr.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  main()
