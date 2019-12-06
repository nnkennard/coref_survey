import sys


#conll12_bc-cctv-00-cctv_0001    0       26      16      12      Chongqing       NNP     (NP)    (GPE)   Luo_huanzhang

with open(sys.argv[1], 'r') as f:
  for line in f:
    if line.strip():
      (doc, part, cluster, sent, sent_start, tokens, pos, parse, ner, speaker) = line.strip().split("\t")
      if parse is not "_":
        print("Span\t" + parse.replace("(", "").replace(")", ""))
      elif len(tokens.split()) == 1:
        print("POS\t" + pos)
      else:
        print("\t".join(["Other", parse, pos, tokens]))



