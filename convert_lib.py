import csv
import json
import nltk

class DatasetName(object):
  conll = 'conll'
  gap = 'gap'
  knowref = 'knowref'
  preco = 'preco'
  red = 'red'
  wikicoref = 'wikicoref'
  ALL_DATASETS = [conll, gap, knowref, preco, red, wikicoref]

class FormatName(object):
  jsonl = 'jsonl'
  bert_jsonl = 'bert_jsonl'
  ALL_FORMATS = [jsonl, bert_jsonl]


NO_SPEAKER = "NO_SPEAKER"

def make_doc_id(dataset, name_tokens):
  return "_".join([dataset] + name_tokens)


def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def add_sentence(curr_doc, curr_sent):
  # This is definitely passed by reference right
  words, speakers = zip(*curr_sent)
  curr_doc.sentences.append(words)
  curr_doc.speakers.append(speakers)


def dataset_from_conll():
  pass

def dataset_from_gap(filename):
  
  dataset = Dataset(DatasetName.gap)

  with open(filename, 'r') as tsvfile:
    for row in csv.DictReader(tsvfile, delimiter='\t'):
      print(row)
      break

def dataset_from_knowref():
  pass

def dataset_from_preco():
  pass

def dataset_from_red():
  pass

def dataset_from_wikicoref(filename):
 
  dataset = Dataset(DatasetName.wikicoref)

  document_counter = 0

  curr_doc = None
  #curr_doc_id = None
  #curr_sent_id = None
  curr_sent = []

  for line in get_lines_from_file(filename):

    if line.startswith("#end") or line.startswith("null"):
      continue
    elif line.startswith("#begin"):
      if curr_doc is not None:
        dataset.documents.append(curr_doc)
      curr_doc_id = make_doc_id(dataset_name, line.split()[3:])
      curr_doc = Document(curr_doc_id)
    else:
      fields = line.split()
      print(fields)
      if not fields:
        if curr_sent:
          add_sentence(curr_doc, curr_sent)
          curr_sent = []
      else:
          word = fields[3]
          curr_sent.append((word, NO_SPEAKER))
  return dataset




class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = []

  def dump_to_jsonl(self):
    return "\n".join(doc.dump_to_jsonl() for doc in self.documents)

  def dump_to_bert_jsonl(self, ):
    pass



class Document(object):
  def __init__(self, doc_id):
    self.doc_id = doc_id
    self.sentences = []
    self.speakers = []
    self.clusters = []


  def dump_to_jsonl(self):
    return json.dumps({
          "document_id": self.doc_id,
          "sentences": self.sentences,
          "speakers": self.speakers,
          "clusters": self.clusters
        })

  def dump_to_bert_jsonl(self, ):
    pass
