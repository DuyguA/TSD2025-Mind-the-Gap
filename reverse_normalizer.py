from entity_utils import tag_all_entities


class ReverseNormalizer(object):
  def __init__(self, normalized_tokens, puncted_tokens):
    self.normalized_tokens = normalized_tokens
    self.puncted_tokens = puncted_tokens
    self.offset_mapping = self.construct_normalized_to_puncted_mapping()
    self.entity_tagger = tag_all_entities
    self.entities = []

  def map_normalized_sentence_to_puncted(self, nstart, nend, puncted_tokens):
    pstart = self.offset_mapping[nstart]
    pend = self.offset_mapping[nend]
    return puncted_tokens[pstart:pend]

  def construct_normalized_to_puncted_mapping(self):
    indices = {}
    sind, eind = 0, 0
    pindex=0
    for ind, nword in enumerate(self.normalized_tokens):
      if nword == '':
        pass
      else:
        indices[sind] = pindex
        eind += nword.strip().count(" ") + 1
      pindex += 1
      sind = eind
    return indices


  def embed_token_tags(self, puncted_tokens, entities):
    if not entities or not puncted_tokens: 
      return puncted_tokens

    tagged_tokens = puncted_tokens[:]
    for entity in entities:
       label = entity['label']
       start_idx = entity['puncted_start']
       end_idx = entity['puncted_end']

       # Insert the opening tag at the start index
       tagged_tokens[start_idx] = f"<{label}>{tagged_tokens[start_idx]}"

       # Insert the closing tag at the end - 1 index
       endt = tagged_tokens[end_idx - 1]
       has_space = endt.endswith(" ")
       endt = endt.strip()
       endt += "</" + label + ">"
       if has_space: endt += " "
       tagged_tokens[end_idx - 1] = endt
    return tagged_tokens

  def reverse_normalize_text(self, ntokens, ptokens):
    entities = self.entity_tagger(ntokens, ptokens)
    r_tokens = self.embed_token_tags(ptokens, entities)
    return r_tokens

  def chunk_text_to_puncted_text(self, chunk_js):
    cstart = chunk["left"]["start_index"] or chunk["mid"]["start_index"] 
    cend = chunk["right"]["end_index"] or chunk["mid"]["end_index"] 
    puncted_tokens = self.map_normalized_sentence_to_puncted(cstart, cend)
    return puncted_tokens

  def reverse_normalize_chunk(self, chunk_js):
    if not self.normalized_tokens or not self.puncted_tokens:
      return chunk_js
    cstart = chunk_js["left"]["start_index"] or chunk_js["mid"]["start_index"] 
    cend = chunk_js["right"]["end_index"] or chunk_js["mid"]["end_index"] 

    rev_tokens = self.reverse_normalize_text(self.normalized_tokens, self.puncted_tokens)
    chunk_ptokens = self.map_normalized_sentence_to_puncted(cstart, cend, rev_tokens)
    chunk_ptext = "".join(chunk_ptokens).strip()
    chunk_js["puncted_text"] = chunk_ptext
    return chunk_js






 

  
