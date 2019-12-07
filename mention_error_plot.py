import sys
import json

def read_file(filename):
  detected_mention_list = []
  with open(filename, 'r') as f:
    for line in f:
      for doc_key, values in json.loads(line.strip()).items():
        detected_mention_list += [(doc_key.replace("/", "-"), mention["start"], mention["end"]) for mention in values]
  return detected_mention_list


def read_data_from_files(data_path):
  detected_mentions = {}
  for model in ["bert_base", "bert_large", "spanbert_base", "spanbert_large"]:
    for span_ratio in ["0.4", "0.8"]:
      filename = "".join([data_path, "/detected-mentions_train_", model, "_conll12_", span_ratio, ".jsonl"])
      detected_mentions[model + "-" + span_ratio] = read_file(filename)

  detected_mentions["bers"] = read_file(data_path + "/bers_mentions_conll-dev.jsonl")
  return detected_mentions

def read_actual_mentions(mention_summary_file, mention_jsonl_file):

  examples = {}
  with open(mention_jsonl_file, 'r') as f:
    for line in f:
      example = json.loads(line.strip())
      examples[example["doc_key"]] = example

  mention_map = {}
  with open(mention_summary_file, 'r') as f:
    for line in f:
      doc, part, entity, sent, start, span, pos, tokens, ner, idk = line.strip().split("\t")
      span_len = len(pos.split())
      if not span.startswith("~"):
        label = (span, None)
      elif span_len == 1:
        label = (None, pos)
      else:
        label = (None, None)
      end = int(start) + span_len - 1
      mention_map[(doc[8:].replace("/", "-") + "_" + part, int(start), end)] = label
  return mention_map
  
      

def main():
  detected_mentions = read_data_from_files(sys.argv[1])
  actual_mentions = read_actual_mentions(sys.argv[2])
  print(actual_mentions)

  for v in detected_mentions.values():

    for mention in v:
      if mention in actual_mentions:
        print(mention)


if __name__ == "__main__":
  main()
