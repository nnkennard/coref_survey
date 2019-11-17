import convert_lib
import conll_lib
import collections
import os

DUMMY_DOC_PART = "0"

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def add_sentence(curr_doc, curr_sent):
  # This is definitely passed by reference right
  words, speakers = zip(*curr_sent)
  curr_doc.sentences.append(words)
  curr_doc.speakers.append(speakers)


def create_dataset(filename):
 
  dataset_name = convert_lib.DatasetName.wikicoref
  dataset = convert_lib.Dataset(dataset_name)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_sent = []
  curr_sent_orig_coref_labels = []
  all_spans = collections.defaultdict(list)


  print(filename)

  for line in get_lines_from_file(filename):

    if line.startswith("#end") or line.startswith("null"):
      continue
    elif line.startswith("#begin"):
      if curr_doc is not None:
        curr_doc.clusters = list(all_spans.values())
        all_spans = collections.defaultdict(list)
        dataset.documents.append(curr_doc)
      curr_doc_id = convert_lib.make_doc_id(dataset_name, line.split()[2:])
      curr_doc = convert_lib.Document(curr_doc_id, DUMMY_DOC_PART)
      sentence_offset = 0
    else:
      fields = line.split()
      if not fields:
        if curr_sent:
          add_sentence(curr_doc, curr_sent)
          coref_spans = conll_lib.get_spans_from_conll(curr_sent_orig_coref_labels,
            sentence_offset)
          all_spans = ldd_append(all_spans, coref_spans)
          sentence_offset += len(curr_sent)
          curr_sent = []
          curr_sent_orig_coref_labels = []
      else:
          word = fields[3]
          coref_label = fields[4]
          curr_sent_orig_coref_labels.append(coref_label)
          curr_sent.append((word, convert_lib.NO_SPEAKER))

  curr_doc.clusters = list(all_spans.values())
  all_spans = collections.defaultdict(list)
  dataset.documents.append(curr_doc)

  return dataset


def convert(data_home):
  output_directory = os.path.join(data_home, "processed", "wikicoref", "test")
  convert_lib.create_processed_data_dir(output_directory)
  test_set = os.path.join(
    data_home, "original", "WikiCoref", "Evaluation", "key-OntoNotesScheme")
  converted_dataset = create_dataset(test_set)
  convert_lib.write_converted(converted_dataset, output_directory)

