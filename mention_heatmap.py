import numpy as np
import pandas as pd
import seaborn as sns
import collections
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def normalize(df):
  result = df.copy()
  for feature_name in df.columns:
    total = df[feature_name].sum()
    result[feature_name] = df[feature_name]/total
  return result

pd.set_option('display.max_columns', 40)

HF_spans = ["Span_(NP)", "Span_(TOP)", "Span_(S)", "Span_(VP)"]
HF_pos = ["POS_PRP$", "POS_PRP"]

def read_file(filename):
  detected_mention_list = []
  with open(filename, 'r') as f:
    for line in f:
      for doc_key, values in json.loads(line.strip()).items():
        detected_mention_list += [(doc_key.replace("/", "-"), mention["start"], mention["end"]) for mention in values]
  return detected_mention_list


MODEL_NAME_MAP = {"bers":"BERS", "spanbert_large-0.4": "E2ESB", "spanbert_large-0.8": "E2ESBr",
                  "Gold": "Gold"}

def read_data_from_files(data_path):
  detected_mentions = {}
  for model, span_ratio in [("spanbert_large", "0.4"), ("spanbert_large", "0.8")]:
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

  mention_map = collections.defaultdict(list)
  with open(mention_summary_file, 'r') as f:
    for line in f:
      doc, part, entity, sent, start, span, pos, tokens, ner, idk = line.strip().split("\t")
      doc_key = doc[8:].replace("/", "-") + "_" + part
      sentence_offset = len(sum(examples[doc_key]["token_sentences"][:int(sent)], []))

      if pos == "DT":
        print(tokens)


      span_len = len(pos.split())
      token_start = sentence_offset + int(start)
      token_end = sentence_offset + int(start) + span_len - 1

      mention = (doc_key, token_start, token_end)

      if span_len == 1:
        mention_map[pos].append(mention)
      elif span.startswith("~"):
        mention_map["MT-NS"].append(mention)
      else:
        mention_map[span].append(mention)
  
  return mention_map

 
def main():

  data_path, gold_summary_file, json_file = sys.argv[1:]

  detected_mentions = read_data_from_files(data_path)
  actual_mentions = read_actual_mentions(gold_summary_file, json_file)

  print(actual_mentions.keys())
  
  heatmap_data = collections.defaultdict(collections.Counter)
  for key, mentions in actual_mentions.items():
    heatmap_data[key]["Gold"]+=len(mentions)

  # False negatives
  for dataset, mention_values in detected_mentions.items():
    dataset_name = MODEL_NAME_MAP[dataset]
    for key, actual_values in actual_mentions.items():
      true_positives = set(actual_values).intersection(set(mention_values))
      heatmap_data[key][dataset_name] += len(true_positives)
      #print(dataset+"\t"+key+"\t"+str(float(len(true_positives))/len(actual_values)))


  frequent_heatmap = {"Other":collections.Counter()}
  frequent =  ["(NP)", "NNP", "PRP", "PRP$"]
  all_keys = heatmap_data.keys()

  for key in all_keys:
    if key in frequent:
      frequent_heatmap[key] = heatmap_data[key]
      del heatmap_data[key]
    else:
      frequent_heatmap["Other"] += heatmap_data[key]
  
  df = pd.DataFrame(heatmap_data)
  hm_df = df.div(df.sum(axis=1), axis=0)
  f_df = pd.DataFrame(frequent_heatmap)
  f_df = f_df.div(f_df.sum(axis=1), axis=0)

  sns.set(font_scale=0.8)

  plt.tight_layout()
  fig3 = plt.figure(constrained_layout=True)
  gs = gridspec.GridSpec(3, 3, figure=fig3)
  ax1 = fig3.add_subplot(gs[0,1])
  ax2 = fig3.add_subplot(gs[1,:])

  ax1.tick_params(rotation=90)
  
  sns.heatmap(f_df, cbar_kws={"shrink": 0.5}, ax=ax1,
                   cmap="GnBu" , xticklabels=True, square=True)
  sns.heatmap(hm_df, cbar_kws={"shrink": 0.5}, ax=ax2,
                   cmap="GnBu" , xticklabels=True, square=True)
  plt.show()
  
if __name__ == "__main__":
  main()
