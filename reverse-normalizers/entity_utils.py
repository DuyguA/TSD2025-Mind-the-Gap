import re
import spacy
from spacy.tokens import Doc, Span





###############
def is_phone_num(word):
  if re.match(r"\d{3}\-\d{3}\-\d{4}", word):
    return True
  if re.match(r"\d{3}\-\d{4}", word):
    return True
  return False

def is_us_phone_num(word):
  if re.match(r"\d{3}\-\d{3}\-\d{4}", word):
    return True
  return False

def need_further_phone_num_parsing(prev_word):
  if prev_word == "+1 ":
    return True
  return False

#########################
  
def is_url(word):
  lwords = ["www.", ".com", ".asp"]
  if any([lw in word for lw in lwords]):
    return True
  return False


def needs_further_url_parsing(puncted_tokens, ent_start):
  if ent_start >= 4:
    http_st = ent_start - 4
    http_token = puncted_tokens[http_st]
    if http_token == "http":
      return True
  return False

#######################

def is_email(word):
  if "@" in word:
    return True
  return False

###################

def contains_digits(word):
  return any([ch.isdigit() for ch in word])

def contains_ampm(word):
  return ("AM" in word or "PM" in word)

def is_time(word):
  return contains_digits(word) and contains_ampm(word) 

#######################

def is_isbn(word, prev_word):
  if "-" in word and prev_word=="ISBN":
    return True
  return False
   
######################


def tag_custom_entity(word, prev_word):
  if is_isbn(word, prev_word):
    return "ISBN"
  if  is_time(word):
    return "TIME"
  if is_phone_num(word):
    return "PHONE_NUM"
  if is_email(word):
    return "EMAIL"
  if is_url(word):
    return "URL"
  return None

########################

def calculate_normalized_token_indice(normalized_tokens, index):
  i=0
  curr_len = 0
  sind, eind = 0, 0
  while i<=index:
    sind = eind
    curr_token = normalized_tokens[i]
    if curr_token == "":
      i += 1
      continue
    curr_len = curr_token.strip().count(" ") + 1
    eind += curr_len
    i += 1
  return sind, eind


###################
def calculate_btw_tokens(normalized_tokens, start, end):
  tokens = normalized_tokens[start:end]
  total_len = sum([len(token.split(" ")) for token in tokens])
  return total_len, " ".join(tokens)



def match_indices_to_normalized(entity_js, normalized_sentence):
  newjs = entity_js.copy()
  pst, pend = entity_js["puncted_start"], entity_js["puncted_end"]
  if pend-pst==1:
    normal_val = normalized_sentence[pst]
    nstart, nend = calculate_normalized_token_indice(normalized_sentence, pst)
    newjs["normalized_val"] = normal_val
    newjs["nstart"] = nstart
    newjs["nend"] = nend
  else:
    nstart, nend = calculate_normalized_token_indice(normalized_sentence, pst)
    btw_len, btw_tokens = calculate_btw_tokens(normalized_sentence, pst, pend)
    newjs["normalized_val"] = btw_tokens
    newjs["nstart"] = nstart
    newjs["nend"] = nstart + btw_len
  return newjs
###################
    

def generate_sentence_char_mapping(puncted_words):
  indices = {}
  curr_start = 0
  for ind, pword in enumerate(puncted_words):
    curr_len = len(pword)
    curr_end = curr_start + curr_len
    indices[curr_start] = ind
    curr_start = curr_end 
  return indices

    

def find_token_indices(start_char, end_char, offset_indices):
  start_ind = offset_indices[start_char]
  if end_char in offset_indices:
    end_ind = offset_indices[end_char]
  else:
    end_ind = offset_indices[end_char+1]
  return start_ind, end_ind
  


#######################
  
def tag_custom_entities(normalized_words, puncted_words):
  entities = []
  for ind, (ns, ps) in enumerate(zip(normalized_words, puncted_words)):
    prev_word = None if ind==0 else puncted_words[ind-1]
    etype =  tag_custom_entity(ps, prev_word)
    ent_start = ind
    ent_end = ind+1
    if etype is not None:
      nstart, nend = calculate_normalized_token_indice(normalized_words, ent_start)

      if etype == "URL" and needs_further_url_parsing(puncted_words, ind):
        ent_start -= 4
        ps = "http://" + ps
        ns = "h t t p forward slash forward slash " + ns 
        nstart -= 8
      elif etype=="PHONE_NUM":
        if need_further_phone_num_parsing(prev_word):
          ent_start -= 1 
          ps = prev_word + ps
          ns = "plus one " + ns
          nstart -= 2
      entities.append({"val": ps, "normalized_val": ns, "label": etype, "puncted_start": ent_start, "puncted_end": ent_end, "nstart": nstart, "nend": nend})
  return entities


def is_numeric_entity(normalized_token, puncted_token):
  if puncted_token.strip() == normalized_token.strip():
    return False
  elif not normalized_token.strip():
    return False
  puncted_token = puncted_token.replace(".", "").replace(",", "").replace("-", "")
  if puncted_token.isdigit():
    return True
  lent = len(puncted_token)
  if lent <= 1:
    return False
  num_digits = sum(char.isdigit() for char in puncted_token)
  if num_digits / lent >= 0.4:
    return True
  return False



def does_ents_intersect(ent1, ent2):
  s1, e1 = ent1["puncted_start"], ent1["puncted_end"]
  s2, e2 = ent2["puncted_start"], ent2["puncted_end"]
  return s1 < e2 and s2 < e1


def merge_two_ents(entity1, entity2):
  ps1, pe1 = entity1["puncted_start"], entity1["puncted_end"]
  ps2, pe2 = entity2["puncted_start"], entity2["puncted_end"]
  len1=pe1-ps1
  len2=pe2-ps2
  merged_ent = entity2 if len2 > len1 else entity1
  return merged_ent


def find_and_merge_intersecting_entities(entities1, entities2):
    merge_candidates = []  # To store pairs of entities to be merged
    used_indices1 = set()  # Track which entities from entities1 are merged
    used_indices2 = set()  # Track which entities from entities2 are merged

    for i, e1 in enumerate(entities1):
        if e1['label'] != "TIME":
            continue  # Skip non-TIME entities
        for j, e2 in enumerate(entities2):
            if e2['label'] != "TIME":
                continue  # Skip non-TIME entities
            if does_ents_intersect(e1, e2):
                merge_candidates.append((e1, e2))
                used_indices1.add(i)
                used_indices2.add(j)

    # Perform merges for intersecting pairs
    merged_entities = []
    for e1, e2 in merge_candidates:
        merged_entities.append(merge_two_ents(e1, e2))

    # Add non-intersecting entities from both lists
    for i, e1 in enumerate(entities1):
        if i not in used_indices1:
            merged_entities.append(e1)

    for j, e2 in enumerate(entities2):
        if j not in used_indices2:
            merged_entities.append(e2)

    return merged_entities


def add_leftover_numeric_entities(entities, normalized_tokens, puncted_tokens):
    new_ents = entities[:]
    num_tokens = len(puncted_tokens)

    is_entity_token = [False] * num_tokens

    for entity in entities:
        start, end = entity['puncted_start'], entity['puncted_end']
        for i in range(start, end):
            if 0 <= i < num_tokens:  # Ensure indices are within bounds
                is_entity_token[i] = True

    # Iterate over tokens and identify numerical entities
    for i, (ntoken, ptoken) in enumerate(zip(normalized_tokens, puncted_tokens)):
        if not is_entity_token[i] and is_numeric_entity(ntoken, ptoken):
            # If the token is not part of any entity and looks like a numerical entity, add it
            estart, eend = i, i+1
            nstart, nend = calculate_normalized_token_indice(normalized_tokens, estart)
            new_ents.append({"val": ptoken, "normalized_val": ntoken, "label": "NUMERIC", "puncted_start": estart, "puncted_end": eend, "nstart": nstart, "nend": nend})

    return new_ents



class EntityTagger:
  def __init__(self):
    self.nlp = spacy.load("en_core_web_trf")

  def tag_all_entities(self, normalized_words, puncted_words):
     if not normalized_words or not puncted_words:
       return []
     custom_ents = tag_custom_entities(normalized_words, puncted_words)
     spacy_ents = self.tag_spacy_entities(normalized_words, puncted_words)
     all_ents = find_and_merge_intersecting_entities(custom_ents, spacy_ents)

     all_ents = add_leftover_numeric_entities(all_ents, normalized_words, puncted_words)
   
     all_ents = sorted(all_ents, key=lambda e: e['puncted_start'])
     return all_ents

  def tag_spacy_entities(self, normalized_words, puncted_words):
    entities = []
    puncted_sent  = "".join(puncted_words)

    doc = self.nlp(puncted_sent)
    ents = doc.ents
    char_offsets = generate_sentence_char_mapping(puncted_words)
  
    #for offset, ind in char_offsets.items():
    #  print(offset, ind, puncted_words[ind])
    for ent in ents:
      val = ent.text
      etype = ent.label_
      start_char = ent.start_char
      end_char = ent.end_char

      if etype == "MONEY":
       ei = ent[0].i
       if ei > 0 :
         prev_word = doc[ei-1]
         if prev_word.text == "$":
           val = "$" + val
           start_char -= 1
      elif etype=="PERCENT":
        val = val.replace(" %", "%")

      try:
        pst, pend = find_token_indices(start_char, end_char, char_offsets)
        ent_js= {"val": val, "label":etype, "puncted_start": pst, "puncted_end": pend}
        new_js = match_indices_to_normalized(ent_js, normalized_words)
        entities.append(new_js)
      except:
        pass
    return entities

