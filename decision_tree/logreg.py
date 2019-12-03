import json
import sys


from sklearn.linear_model import LogisticRegressionCV


import pandas as pd

def get_mentions(data_frame):
  mentions = set()
  for _, example in data_frame.iterrows():
    mentions.add((example["doc_key"], 
                  example["ant_start"],
                  example["ant_end"]))
    mentions.add((example["doc_key"], 
                  example["con_start"],
                  example["con_end"]))
  return sorted(list(mentions))


def flatten_parse_spans(parse_spans, token_sentences):
  overall_parse_span_map = {}
  token_count = 0
  for sentence, sentence_parse_spans in zip(token_sentences, parse_spans):
    for start, end, label in sentence_parse_spans:
      overall_span = token_count + start, token_count + end
      assert overall_span not in overall_parse_span_map
      overall_parse_span_map[overall_span] = label.replace("*", "")
    token_count += len(sentence)

  return overall_parse_span_map




def get_features(mentions, data_file):
  docs = {}
  with open(data_file, 'r') as f:
    for line in f:
      obj = json.loads(line.strip())
      docs[obj["doc_key"]] = obj

  features = []
  for doc_key, start, end in mentions:

    doc = docs[doc_key]
    parse_spans = flatten_parse_spans(doc["parse_spans"], doc["token_sentences"])
    features.append([end - start, 
                     parse_spans.get((start, end), "-"),
                     sum(doc["pos"], [])[start:end+1]])

  for i in features:
    print(i)
  print(len(features))


  

def main():

  data_names = ["doc_key", "ant_start", "ant_end", "con_start", "con_end", "label"]
  data_frame = pd.read_csv(sys.argv[1], sep="\t", names=data_names)
  mentions = get_mentions(data_frame)
  get_features(mentions, sys.argv[2])
  pass


if __name__ == "__main__":
  main()
