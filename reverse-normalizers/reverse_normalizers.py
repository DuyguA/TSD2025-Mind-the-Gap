class ReverseNormalizer(object):
  def __init__(self, ent_tagger_obj, normalized_tokens, puncted_tokens):
    self.normalized_tokens = normalized_tokens
    self.puncted_tokens = puncted_tokens
    self.offset_mapping = self.construct_normalized_to_puncted_mapping()
    self.entity_tagger = ent_tagger_obj
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
    self.entities = self.entity_tagger.tag_all_entities(ntokens, ptokens)
    r_tokens = self.embed_token_tags(ptokens, self.entities)
    return r_tokens

  def find_ent_on_boundary(self, start, end, next_start):
    for entity in self.entities:
      ent_nstart, ent_nend = entity["nstart"], entity["nend"]
      if start <= ent_nstart < end  and next_start <= ent_nend:
        return entity
    return None

  def shift_boundary(self, lstart, lend, mstart):
    boun_ent = self.find_ent_on_boundary(lstart, lend, mstart)
    lpend = boun_ent["puncted_start"]-1 # finish left on the start of the ent, start mid from there
    mpstart = lpend + 1
    return lpend, mpstart


  def calculate_window_boundary(self, lstart, lend, mstart, side="left"):
    if side == "left":
      lpstart = self.offset_mapping[lstart]
      if self.window_needs_shifting(mstart):
        print("LEFT TO MID SHIFT")
        lpend, mpstart = self.shift_boundary(lstart, lend, mstart)
      else:
        mpstart = self.offset_mapping[mstart]
        lpend = mpstart-1
      return lpstart, lpend, mpstart
    elif side=="right":
      if self.window_needs_shifting(mstart):
        print("MID TO RIGHT SHIFT")
        lpend, mpstart = self.shift_boundary(lstart, lend, mstart)
      else:
        mpstart = self.offset_mapping[mstart]
        lpend = mpstart-1
      return lpend, mpstart

  def window_needs_shifting(self, mstart):
    return mstart not in self.offset_mapping

  def is_window_empty(self, nstart):
    return nstart is None

  def segment_puncted_text_into_windows(self, left_window, mid_window, right_window, puncted_tokens):
    lstart, lend = left_window["start_index"], left_window["end_index"]
    mstart, mend = mid_window["start_index"], mid_window["end_index"]
    rstart, rend = right_window["start_index"], right_window["end_index"]

    ltokens = left_window["tokens"]
    mtokens = mid_window["tokens"]
    rtokens = right_window["tokens"]
    #print(ltokens, mtokens, rtokens)


    # fix left-mid boundary
    # fix left end and mid start, calculate left text
    if self.is_window_empty(lend):
      lplen, lptext = 0, ""
      mpstart = self.offset_mapping[mstart]
    else:
      #print("Here in left")
      lpstart, lpend, mpstart = self.calculate_window_boundary(lstart, lend, mstart, side="left")
      lplen = lpend - lpstart+1
      lptokens = puncted_tokens[:lplen]
      lptext = "".join(lptokens).strip()

    # fix mid-right boundary
    # fix mid end and right start, calculate mid and right texts
    if self.is_window_empty(rstart):
      rptext = ""
      mptokens = puncted_tokens[lplen:]
      mptext = "".join(mptokens)
    else:
      #print("Here in right")
      #print(mstart, mend, rstart, "all indices")
      mpend, rpstart = self.calculate_window_boundary(mstart, mend, rstart, side="right")
      rpend = self.offset_mapping[rend]
      #print(mpstart, mpend, rpstart, "new indices")
      #for ind, puncted_token in enumerate(puncted_tokens):
      #  print(ind, puncted_token)
      mplen = mpend-mpstart +1 
      mptokens = puncted_tokens[lplen:lplen+mplen]
      #print(mplen, mptokens)
      #for ind, puncted_token in enumerate(mptokens):
      #  print(ind, puncted_token)
      rptokens = puncted_tokens[lplen+mplen:]

      mptext = "".join(mptokens).strip()
      rptext = "".join(rptokens).strip()
    segmented_text = lptext + "|" + mptext + "|" + rptext
    #print(lptext, "|", mptext , "|",  rptext)
    #print(puncted_tokens)
    #print("==================================")
    return segmented_text

    

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

    #print(self.normalized_tokens)
    #print(self.puncted_tokens)
    
    #for offset, ind in self.offset_mapping.items():
    #  print(offset, ind, self.normalized_tokens[ind], self.puncted_tokens[ind])
    #print("==========================")



    rev_tokens = self.reverse_normalize_text(self.normalized_tokens, self.puncted_tokens)
    chunk_ptokens = self.map_normalized_sentence_to_puncted(cstart, cend, rev_tokens)
    chunk_ptext = "".join(chunk_ptokens).strip()
    chunk_js["puncted_text"] = chunk_ptext
 
    lwindow, mwindow, rwindow = chunk_js["left"], chunk_js["mid"], chunk_js["right"]
    sptext = self.segment_puncted_text_into_windows(lwindow, mwindow, rwindow, chunk_ptokens)
    chunk_js["segmented_puncted_text"] = sptext
    return chunk_js

