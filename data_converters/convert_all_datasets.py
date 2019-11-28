import sys
import gap_converter
import ontonotes_converter
import preco_converter
import wikicoref_converter

def main():
  data_home = sys.argv[1]
  print("GAP")
  gap_converter.convert(data_home) 
  print("Preco")
  preco_converter.convert(data_home) 
  print("Wikicoref")
  wikicoref_converter.convert(data_home) 
  print("Ontonotes")
  ontonotes_converter.convert(data_home) 


if __name__ == "__main__":
  main()
