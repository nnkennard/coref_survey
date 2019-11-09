import os
import json
import convert_lib
import tqdm

PRECO = convert_lib.DatasetName.preco
DUMMY_DOC_PART = '0'

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()


def create_dataset(filename):

  dataset = convert_lib.Dataset(PRECO)

  lines = get_lines_from_file(filename)
  for line in tqdm.tqdm(lines):
    orig_document = json.loads(line)
    new_document = convert_lib.Document(
        convert_lib.make_doc_id(PRECO, orig_document["id"]), DUMMY_DOC_PART)
    sentence_offsets = []
    token_count = 0
    new_document.sentences = []
    for sentence in orig_document["sentences"]:

      sentence_offsets.append(token_count)
      token_count += len(sentence)

      maybe_sentence = [str(token) for token in sentence]
      new_document.sentences.append(maybe_sentence)
    new_document.speakers = convert_lib.make_empty_speakers(new_document.sentences)
    new_document.clusters = []
    for cluster in orig_document["mention_clusters"]:
        new_cluster = []
        for sentence, begin, end in cluster:
          new_cluster.append([sentence_offsets[sentence] + begin,
          sentence_offsets[sentence] + end - 1])
        new_document.clusters.append(new_cluster)
    dataset.documents.append(new_document)

  return dataset

def convert(data_home):
  preco_directory = os.path.join(data_home, "original", "PreCo_1.0")
  output_directory = os.path.join(data_home, "processed", PRECO)
  preco_datasets = {}
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev]:
    print(split)
    input_filename = os.path.join(preco_directory, split + "." +
        convert_lib.FormatName.jsonl)
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
