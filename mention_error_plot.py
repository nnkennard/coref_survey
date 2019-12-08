import numpy as np
import pandas as pd
import seaborn as sns
import collections
import sys
import json
import matplotlib.pyplot as plt

def normalize(df):
  result = df.copy()
  for feature_name in df.columns:
    total = df[feature_name].sum()
    result[feature_name] = df[feature_name]/total
  return result

pd.set_option('display.max_columns', 20)


HIGH_FREQ = ["Span_(TOP)", "POS_VBD", "POS_VB", "Span_(NML)", "POS_NNP", #"POS_PRP$", "Span_(NP)"
]

MED_FREQ = ["Span_(ADVP)", "Span_(ADJP)", "POS_JJ", "POS_CD", "POS_PRP", "POS_NN", "Span_(VP)", "Span_(S)", "Other_", "POS_VBP", "POS_VBZ", "POS_VBG", "POS_VBN"]

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
      doc_key = doc[8:].replace("/", "-") + "_" + part
      sentence_offset = len(sum(examples[doc_key]["token_sentences"][:int(sent)], []))
      span_len = len(pos.split())
      if not span.startswith("~"):
        label = (span, None)
      elif span_len == 1:
        label = (None, pos)
      else:
        label = (None, None)
      token_start = sentence_offset + int(start)
      token_end = sentence_offset + int(start) + span_len - 1
      mention_map[(doc_key , token_start, token_end)] = label
  return mention_map
  
def get_key_from_details(span, pos):    
  if span is not None:
    key = "Span_" + span
  elif pos is not None:
    key = "POS_" + pos
  else:
    key = "Other"
  return key

def main():
  detected_mentions = read_data_from_files(sys.argv[1])
  actual_mentions = read_actual_mentions(sys.argv[2], sys.argv[3])

  gold_map = {}
  true_positive_map = collections.defaultdict(lambda : collections.Counter())
  false_negative_map = collections.defaultdict(lambda : collections.Counter())

  # True positives
  for dataset, mention_values in detected_mentions.items():
    for mention in mention_values: 
      details = actual_mentions.get(mention, None)
      if details is not None:
        key = get_key_from_details(*details)
        if key in MED_FREQ:
          true_positive_map[dataset][key] += 1

  # False negatives
  for dataset, mention_values in detected_mentions.items():
    false_negatives = set(actual_mentions.keys()) - set(mention_values)
    for mention in false_negatives:
      key = get_key_from_details(*actual_mentions[mention])
      if key in MED_FREQ:
        false_negative_map[dataset][key] += 1
        

  true_positive_df = normalize(pd.DataFrame(true_positive_map))
  

  #false_negative_df = normalize(pd.DataFrame(false_negative_map))
  false_negative_df = pd.DataFrame(false_negative_map)

  from matplotlib.colors import ListedColormap

  sns.set()

  false_negative_df.T.plot(kind="bar", stacked=True)
  plt.show()


if __name__ == "__main__":
  main()
