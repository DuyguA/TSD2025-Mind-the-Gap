import json


'''
key = "puncted_text"
ofilen = "puncted_chunks.json"
'''
outdict = {}


key = "segmented_puncted_text"
ofilen = "segmented_puncted_chunks.json"

with open("corrected_new.jsonl", "r") as injs:
  for line in injs:
    linejs = json.loads(line)
    chunks = linejs["chunks"]
    for chunk in chunks:
      newchunk = {}

      newchunk["text"] = chunk[key]
      left, mid, right = chunk["left"], chunk["mid"], chunk["right"]
      start_time = left["start_time"] or mid["start_time"]
      end_time = right["end_time"] or mid["end_time"]

      lstart_time = left["start_time"] 
      lend_time = left["end_time"] 

      mstart_time = mid["start_time"] 
      mend_time = mid["end_time"] 

      rstart_time = right["start_time"] 
      rend_time = right["end_time"] 


      newchunk["start_time"] = start_time
      newchunk["end_time"] = end_time

      newchunk["lstart_time"] = lstart_time
      newchunk["lend_time"] = lend_time

      newchunk["mstart_time"] = mstart_time
      newchunk["mend_time"] = mend_time

      newchunk["rstart_time"] = rstart_time
      newchunk["rend_time"] = rend_time

      cid = chunk["global_chunk_number"]
      outdict[cid] = newchunk


with open(ofilen, "w") as ofile:
  outjs = json.dumps(outdict)
  ofile.write(outjs)
