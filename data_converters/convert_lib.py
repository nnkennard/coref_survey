import tqdm
import json

import collections
import csv
import json
import os

from bert import tokenization
VOCAB_FILE = "/home/nnayak/spanbert-coref-fork/cased_config_vocab/vocab.txt"
MAX_SEGMENT_LEN = 512
TOKENIZER = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

class DatasetName(object):
  conll12 = 'conll12' # TODO delete one of these later
  gap = 'gap'
  knowref = 'knowref'
  preco = 'preco'
  red = 'red'
  wikicoref = 'wikicoref'
  ALL_DATASETS = [conll12, gap, knowref, preco, red, wikicoref]

class DatasetSplit(object):
  train = 'train'
  test = 'test'
  dev = 'dev'
  valid = 'valid'

class FormatName(object):
  jsonl = 'jsonl'
  file_per_doc = 'file_per_doc'
  txt = 'txt'
  ALL_FORMATS = [jsonl, file_per_doc, txt]

def get_filename(data_home, dataset_name, dataset_split, format_name):
  return os.path.join(data_home, 'processed', dataset_name,
      dataset_split + "." + format_name)

def create_processed_data_dir(path):
  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s " % path)

NO_SPEAKER = "-"

def make_doc_id(dataset, doc_name):
  if type(doc_name) == list:
    doc_name = "_".join(doc_name)
  return "_".join([dataset, doc_name])


def make_empty_speakers(sentences):
  return [[NO_SPEAKER for token in sent] for sent in sentences]

CLS = "[CLS]"
SPL = "[SPL]"
SEP = "[SEP]"


def create_maps(subword_list, segment_idx, running_token_idx, speaker):
    subword_to_word = [[local_token_idx + running_token_idx]* len(token_subwords)
                            for local_token_idx, token_subwords in enumerate(subword_list)]
    running_token_idx += len(subword_list)

    subword_to_word_flat = sum(subword_to_word, [])
    subword_to_word_flat = [0] + subword_to_word_flat + [subword_to_word_flat[-1]]
    subword_list_flat = [CLS] + sum(subword_list, []) + [SEP]
    subword_to_segment = [segment_idx] * len(subword_list_flat)

    speaker_list = [""] * len(subword_to_segment)    

    return (subword_list_flat, speaker_list, subword_to_segment, subword_to_word_flat, running_token_idx)


_maybe_unused = """
def subdivide_sentence_by_segment_length(tokens, max_segment_len):
    segments = []
    subtoken_list = [tokenizer.tokenize(token) for token in tokens]
    curr_segment = []
    curr_segment_subtokens = 0
    for token_subtokens in subtoken_list:
        if curr_segment_subtokens + len(token_subtokens) > max_segment_len:
            segments.append(curr_segment)
            curr_segment = [token_subtokens]
            curr_segment_subtokens = len(token_subtokens)
        else:
            curr_segment.append(token_subtokens)
            curr_segment_subtokens += len(token_subtokens)
    segments.append(curr_segment)
    return segments
"""


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
    create_processed_data_dir(directory)
    for doc in tqdm.tqdm(self.documents):
      with open(directory + "/" + doc.doc_id + "_" + doc.doc_part + ".txt", 'w') as f:
        f.write("\n".join(doc.dump_to_stanford()))

class Document(object):
  def __init__(self, doc_id, doc_part):
    self.doc_id = doc_id
    self.doc_part = doc_part
    self.doc_key = "UNK"
    self.sentences = []
    self.speakers = []
    self.clusters = []
    self.subtoken_map = None
    self.sentence_map = None
    self.token_sentences = None
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

  def _subdivide_sentence_by_segment_length(self, tokens):
      segments = []
      subtoken_list = [TOKENIZER.tokenize(token) for token in tokens]
      curr_segment = []
      curr_segment_subtokens = 0
      for token_subtokens in subtoken_list:
          if curr_segment_subtokens + len(token_subtokens) > MAX_SEGMENT_LEN:
              segments.append(curr_segment)
              curr_segment = [token_subtokens]
              curr_segment_subtokens = len(token_subtokens)
          else:
              curr_segment.append(token_subtokens)
              curr_segment_subtokens += len(token_subtokens)
      segments.append(curr_segment)
      return segments


  def calculate_subtokens(self):
    assert self.sentences
    sentences = []
    new_speakers = []
    sentence_map = []
    subtoken_map = []
    running_token_idx = 0
    segment_idx = 0
    for sentence, speakers in zip(self.sentences, self.speakers):
      print("Sentence: ", sentence)
      segments = self._subdivide_sentence_by_segment_length(sentence)
      assert len(set(speakers)) == 1
      speaker, = set(speakers)
      for segment in segments:
          (
              subword_list_flat,
              speaker_list,
              subword_to_segment,
              subword_to_word_flat,
              running_token_idx) = create_maps(segment, segment_idx, running_token_idx, speaker)
          sentences.append(subword_list_flat)
          new_speakers.append(speaker_list)
          sentence_map += subword_to_segment
          subtoken_map.append(subword_to_word_flat)
          segment_idx += 1

    self.token_sentences = self.sentences
    self.sentences = sentences
    self.sentence_map = sentence_map
    self.subtoken_map = subtoken_map
    self.speakers = new_speakers


  def dump_to_jsonl(self):
    if self.subtoken_map is None:
      self.calculate_subtokens()
    return [json.dumps({
          "doc_key": self.doc_key,
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": self.sentences,
          "sentence_map": self.sentence_map,
          "subtoken_map": self.subtoken_map,
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
    dataset.dump_to_stanford(prefix + "-fpd/")
 
