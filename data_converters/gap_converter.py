import convert_lib
import os
import csv
import collections

from nltk.tokenize import sent_tokenize, word_tokenize


GAP = convert_lib.DatasetName.gap
DUMMY_DOC_PART = "0"

def get_mention_char_indices(row):
  index_map = {}
  for mention in ["A", "B", "Pronoun"]:
    mention_start = int(row[mention + "-offset"])
    mention_text = row[mention]
    index_map[mention] = (mention_start, mention_start + len(mention_text))
  return index_map


def clean_up_text(row):
  new_text = row["Text"]
  mentions = get_mention_char_indices(row)
  for mention, (start, end) in mentions.items():
    for pos in [start -  1, end]:
      if new_text[pos] != " ":
        if new_text[pos] in "/.-":
          if pos == len(new_text) - 2:
            continue
          new_text = new_text[:pos] + ' ' + new_text[pos+1:]
  new_text = new_text.replace("''", "``").replace("'", "`") # TODO: both necessary?
  return new_text


def char_to_tok_idx(text, char_indices):
  tokens = sum([word_tokenize(sent) for sent in sent_tokenize(text)], [])

  index_map = {}

  curr_token_start = 0
  for i, (curr_token, next_token) in enumerate(zip(tokens, tokens[1:] + [None])):
    index_map[curr_token_start] = i
    curr_token_start += len(curr_token)
    if next_token is None:
      break
    while text[curr_token_start:curr_token_start + len(next_token)] != next_token:
      curr_token_start += 1
    assert text[curr_token_start:curr_token_start + len(next_token)] == next_token


  token_indices  = []

  for mention_text, str_char_offset in char_indices:
    char_offset = int(str_char_offset)
    end_offset = None
    start_token = index_map[char_offset]
    min_end_offset = char_offset + len(mention_text)
    while min_end_offset not in index_map:
      min_end_offset += 1
      if min_end_offset == len(text):
        end_offset = len(tokens)
        break

    if end_offset is None:
      assert char_offset in index_map
      end_offset = index_map[min_end_offset]

    token_indices.append((start_token, end_offset - 1))

  return token_indices


def create_dataset(filename):
  dataset = convert_lib.Dataset(GAP)
  with open(filename, 'r') as tsvfile:
    for row in csv.DictReader(tsvfile, delimiter='\t'):
      text = clean_up_text(row)
      
      curr_document = convert_lib.Document(convert_lib.make_doc_id(GAP, row["ID"]), DUMMY_DOC_PART)
      (pronoun_indices, a_indices, b_indices) = char_to_tok_idx(text,
        ((row["Pronoun"], row["Pronoun-offset"]), (row["A"], row["A-offset"]),
          (row["B"], row["B-offset"])))
      true_cluster = [pronoun_indices]
      other_cluster = []
      if row["A-coref"] == "TRUE":
        true_cluster.append(a_indices)
      else:
        other_cluster.append(a_indices)
      if row["B-coref"] == "TRUE":
        true_cluster.append(b_indices)
      else:
        other_cluster.append(b_indices)
      curr_document.sentences = [
          word_tokenize(sent) for sent in sent_tokenize(text)]

      tok_running_count = 0
      char_running_count = 0
      for i, sent in enumerate(curr_document.sentences):
        for j, tok in enumerate(sent):
          char_running_count += len(tok) + 1
        tok_running_count += len(sent)


      curr_document.speakers = convert_lib.make_empty_speakers(curr_document.sentences)
      curr_document.clusters = [true_cluster, other_cluster]
      dataset.documents.append(curr_document)

  return dataset


def convert(data_home):
  gap_directory = os.path.join(data_home, "original", "GAP", "gap-coreference")
  output_directory = os.path.join(data_home, "processed", "gap")

  gap_datasets = {}

  for split, split_name in zip(
      [convert_lib.DatasetSplit.dev, convert_lib.DatasetSplit.valid,
        convert_lib.DatasetSplit.test], ["development", "validation", "test"]):
    input_filename = os.path.join(gap_directory, "gap-" + split_name + ".tsv")
    print(input_filename)
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)

