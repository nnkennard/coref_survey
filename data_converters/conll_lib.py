import collections
def coref_to_spans(coref_col, offset):
  span_starts = collections.defaultdict(list)
  complete_spans = []
  for i, orig_label in enumerate(coref_col):
    if orig_label == '-':
      continue
    else:
      labels = orig_label.split("|")
      for label in labels:
        if label.startswith("("):
          if label.endswith(")"):
            complete_spans.append((i, i, label[1:-1]))
          else:
            span_starts[label[1:]].append(i)
        elif label.endswith(")"):
          ending_cluster = label[:-1]
          assert len(span_starts[ending_cluster]) in [1, 2]
          maybe_start_idx = span_starts[ending_cluster].pop(-1)
          complete_spans.append((maybe_start_idx, i, ending_cluster))

  span_dict = collections.defaultdict(list)
  for start, end, cluster in complete_spans:
    span_dict[cluster].append((offset + start, offset + end))


  return span_dict

def split_parse_label(label):
  curr_chunk = ""
  chunks = []
  for c in label:
    if c in "()":
      if curr_chunk:
        chunks.append(curr_chunk)
      curr_chunk = c
    else:
      curr_chunk += c
  chunks.append(curr_chunk)
  return chunks


def parse_to_spans(parse_col):
  span_starts = collections.defaultdict(list)
  stack = []
  label_map = {}
  for i, orig_label in enumerate(parse_col):
    labels = split_parse_label(orig_label)
    for label in labels:
      if label.endswith(")"):
        span_start, start_idx = stack.pop(0)
        assert (span_start, i) not in label_map
        label_map[(start_idx, i)] = span_start + label
      elif label.startswith("("):
        stack.insert(0, [label, i])
      else:
        stack[0][0] += label
  return label_map
  #return [(k[0], k[1], v) for k,v in label_map.items()]


