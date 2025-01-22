from lxml import etree

# Parse the XML file

def parse_xml(fname):
  puncted_sentences = []
  normalized_sentences = []
  timestamps = []

  tree = etree.parse(fname)

  # Find all paragraphs
  paragraphs = tree.xpath("//p")


  for paragraph in paragraphs:

    sentences = paragraph.xpath("./s")  # Find all <s> within the paragraph
    for sentence in sentences:
        try:
          psent, nsent, tstamps = [], [], []

          tokens = sentence.xpath("./t")  # Find all <t> within the sentence
          for token in tokens:
            # If <t> has text, extract it (e.g., multiword entities like "12.5")
            token_text = token.text.strip() if token.text else None
            if token_text in ".,;:?!)":
              psent[-1] = psent[-1].rstrip()
            elif token_text == "$":
              continue

            if token_text == "(":
              psent.append(token_text)
            else:
              psent.append(token_text+ " ")

            # Extract nested <n> elements within <t>
            nested_elements = token.xpath("./n")

            prons = []
            for nested in nested_elements:
                # Extract attributes of each <n>
                pronunciation = nested.get("pronunciation")
                pronunciation = pronunciation.lstrip("_")
                prons.append(pronunciation) 
                start = nested.get("start")
                end = nested.get("end")
                start = int(start) if start is not None else start
                end = int(end) if end is not None else end
                tstamps.append((start, end))
            real_pron = " ".join(prons)
            nsent.append(real_pron)
            if real_pron.endswith((" dollar", " dollars")) and "$" not in token_text:
              psent[-1] = "$" + psent[-1]
          psent[-1] = psent[-1].strip()
          normalized_sentences.append(nsent)
          puncted_sentences.append(psent)
          timestamps.append(tstamps)
        except:
          pass

  return normalized_sentences, puncted_sentences, timestamps

def construct_normalized_to_puncted_mapping(normalized_tokens):
  indices = {}
  sind, eind = 0, 0
  pindex=0
  for ind, nword in enumerate(normalized_tokens):
    if nword == '':
      pass
    else:
      indices[sind] = pindex
      eind += nword.strip.count(" ") + 1
    pindex += 1
    sind = eind
  return indices


def construct_normalized_to_puncted_mapping(normalized_tokens):
  indices = {}
  sind, eind = 0, 0
  pindex=0
  for ind, nword in enumerate(normalized_tokens):
    if nword == '':
      pass
    else:
      indices[sind] = pindex
      eind += nword.strip().count(" ") + 1
    pindex += 1
    sind = eind
  return indices



'''
fname = "examples/date2.xml"
nsents, psents, tstamps = parse_xml(fname)


nsent, psent, tstamp = nsents[0], psents[0], tstamps[0]


for ind, (n, p) in enumerate(zip(nsent, psent)):
    print(ind, p, n)

print("================")

indices = construct_normalized_to_puncted_mapping(nsent)
nsent = [word for phrase in nsent for word in phrase.split()]
for ind, pind in indices.items():
  print(ind, pind, nsent[ind], psent[pind])

print(len(nsent), len(psent), len(tstamp))

print(tstamp)
print(nsent)


print("==================")
print("".join(psent))
print("==================")


from entity_utils import tag_custom_entities, tag_spacy_entities, tag_all_entities
ents = tag_all_entities(nsent, psent)

print(ents)
'''
