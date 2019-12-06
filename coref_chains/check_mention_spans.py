import sys

with open(sys.argv[1], 'r') as f:
  for line in f:
    if line.strip():
      (doc, part, cluster, sent, sent_start, tokens, pos, parse, ner, speaker) = line.strip().split("\t")
      if parse is not "_":
        print("Span\t" + parse.replace("(", "").replace(")", ""))
      elif len(tokens.split()) == 1:
        print("POS\t" + pos)
      else:
        print("\t".join(["Other", doc, parse, pos, tokens]))



