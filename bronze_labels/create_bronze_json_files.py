import collections
import json
import sys

MODEL_TO_FILENAME = {
  "bert_base": (128, "detected-mentions_train_bert_base_conll12_0.4.jsonl"),
  "bert_large": (384, "detected-mentions_train_bert_large_conll12_0.4.jsonl"),
  "spanbert_base": (384, "detected-mentions_train_spanbert_base_conll12_0.4.jsonl"),
  "spanbert_large": (512, "detected-mentions_train_spanbert_large_conll12_0.4.jsonl"),
}

WINDOW_TO_FILENAME ={
  128: "dev.english.128.jsonlines",
  384: "dev.english.384.jsonlines",
  512: "dev.english.512.jsonlines"
}

def get_bronze_mentions(clusters, predicted_mentions):
  gold_mentions = set(tuple(i) for i in sum(clusters, []))
  print(gold_mentions)
  high_precision = sorted(list(
    predicted_mentions.intersection(gold_mentions)))
  high_recall = sorted(list(
    predicted_mentions.union(gold_mentions)))
  return high_precision, high_recall

def get_examples(data_path, window):
  jsonl_file = data_path + "/" + WINDOW_TO_FILENAME[window]
  examples = collections.OrderedDict()
  with open(jsonl_file, 'r') as f:
    for line in f:
      example = json.loads(line.strip())
      examples[example["doc_key"]] = example
  return examples



def main():
  predictions_data_path, = sys.argv[1:]
   
  for model, (window, filename) in MODEL_TO_FILENAME.items():

    examples = get_examples(predictions_data_path, window)

    predictions = {}
    with open(predictions_data_path + "/" + filename, 'r') as f:
      for line in f:
        obj = json.loads(line.strip())
        for k, v in obj.items():
          predictions[k] = [(span["start"], span["end"]) for span in v]
    
    high_p_filename = filename.replace(
        ".jsonl", "_" + model + "_highp.jsonl")
    high_r_filename = filename.replace(
        ".jsonl", "_" + model + "_highr.jsonl")

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
