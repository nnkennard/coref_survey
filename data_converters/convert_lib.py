import tqdm
import json

import collections
import csv
import json
import os

class DatasetName(object):
  ontonotes = 'ontonotes'
  conll = 'conll' # TODO delete one of these later
  gap = 'gap'
  knowref = 'knowref'
  preco = 'preco'
  red = 'red'
  wikicoref = 'wikicoref'
  ALL_DATASETS = [ontonotes, conll, gap, knowref, preco, red, wikicoref]

class DatasetSplit(object):
  train = 'train'
  test = 'test'
  dev = 'dev'
  valid = 'valid'

class FormatName(object):
  jsonl = 'jsonl'
  jsonlb = 'jsonlb'
  file_per_doc = 'file_per_doc'
  txt = 'txt'
  ALL_FORMATS = [jsonl, jsonlb, file_per_doc, txt]

def get_filename(data_home, dataset_name, dataset_split, format_name):
  return os.path.join(data_home, 'processed', dataset_name,
      dataset_split + "." + format_name)

NO_SPEAKER = "-"

def make_doc_id(dataset, doc_name):
  if type(doc_name) == list:
    doc_name = "_".join(doc_name)
  return "_".join([dataset, doc_name])


def make_empty_speakers(sentences):
  return [[NO_SPEAKER for token in sent] for sent in sentences]

#def dataset_from_preco(filename):
#
#  dataset = Dataset(DatasetName.preco)
#  
#  for line in get_lines_from_file(filename):
#    orig_document = json.loads(line)
#    new_document = Document(
#        make_doc_id(DatasetName.preco, orig_document["id"]))
#    sentence_offsets = []
#    token_count = 0
#    new_document.sentences = []
#    for sentence in orig_document["sentences"]:
#      sentence_offsets.append(token_count)
#      token_count += len(sentence)
#      new_document.sentences.append([str(token.encode('utf-8')) for token in sentence])
#    new_document.speakers = make_empty_speakers(new_document.sentences)
#    new_document.clusters = []
#    for cluster in orig_document["mention_clusters"]:
#      new_cluster = []
#      for sentence, begin, end in cluster:
#        new_cluster.append([sentence_offsets[sentence] + begin,
#          sentence_offsets[sentence] + end])
#      new_document.clusters.append(new_cluster)
#    dataset.documents.append(new_document)
#
#  return dataset

class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = []

  def _dump_lines(self, function, file_handle):
    lines = []
    for doc in self.documents:
      lines += doc.apply_dump_fn(function)
    file_handle.write("\n".join(lines))

  def dump_to_mconll(self, file_handle):
    self._dump_lines("mconll", file_handle)

  def dump_to_jsonl(self, file_handle):
    self._dump_lines("jsonl", file_handle)

  def dump_to_stanford(self, directory):
    if not os.path.exists(directory):
      os.makedirs(directory)
    for doc in tqdm.tqdm(self.documents):
      with open(directory + "/" + doc.doc_id + "_" + doc.doc_part + ".auto_conll", 'w') as f:
        f.write("\n".join(doc.dump_to_stanford()))

class Document(object):
  def __init__(self, doc_id, doc_part):
    self.doc_id = doc_id
    self.doc_part = doc_part
    self.sentences = []
    self.speakers = []
    self.clusters = []
    self.FN_MAP = {
      "mconll": self.dump_to_mconll,
      "jsonl": self.dump_to_jsonl,
      "file_per_doc": self.dump_to_stanford}



  def apply_dump_fn(self, function):
    return self.FN_MAP[function]()

  def _get_conll_coref_labels(self):
    coref_labels = collections.defaultdict(list)
    for cluster, tok_idxs in enumerate(self.clusters):
      for tok_start, tok_end in tok_idxs:
        if tok_start == tok_end:
          coref_labels[tok_start].append("({})".format(cluster))
        else:
          coref_labels[tok_start].append("({}".format(cluster))
          coref_labels[tok_end].append("{})".format(cluster))

    return coref_labels

  def dump_to_jsonl(self):
    return [json.dumps({
          "doc_key": "nw",
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": self.sentences,
          "speakers": self.speakers,
          "clusters": self.clusters
        })]

  def dump_to_stanford(self):
    return [" ".join(sentence) for sentence in self.sentences]

     
  def dump_to_mconll(self):
    document_name = self.doc_id
    coref_labels = self._get_conll_coref_labels()
    sent_start_tok_count = 0

    mconll_lines = ["#begin document " + document_name + "\n"] 

    for i_sent, sentence in enumerate(self.sentences):
      for i_tok, token in enumerate(sentence):
        coref_label_vals = coref_labels.get(sent_start_tok_count + i_tok)
        if not coref_label_vals:
          label = '-'  
        else:
          label = "|".join(coref_label_vals)
        mconll_lines.append("\t".join([
          self.doc_id, self.doc_part, str(i_tok), token, self.speakers[i_sent][i_tok], label]))
      sent_start_tok_count += len(sentence)
      mconll_lines.append("")

    mconll_lines.append("\n#end document " + document_name + "\n")

    return mconll_lines


def write_converted(dataset, prefix):
    print(prefix)
    with open(prefix + ".mconll", 'w') as f:
      dataset.dump_to_mconll(f)
    with open(prefix + ".jsonl", 'w') as f:
      dataset.dump_to_jsonl(f)
    dataset.dump_to_stanford(prefix + "-stanford/")
 
