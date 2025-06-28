from file_parser import parse_xml
from reverse_normalizer import ReverseNormalizer
from entity_utils import EntityTagger

finished_files = open("completed_files.txt", "r").read().split("\n")
finished_files = finished_files[-1]


import os
import json  # Assuming files are in JSON format

etagger = EntityTagger()


def chunk_single_file(file_path, left_window_ms=5000, mid_window_ms=20000, right_window_ms=5000):
    normalized_tokens, puncted_tokens , timestamps =  parse_xml(file_path)

    all_ntokens =  [li for subli in normalized_tokens for li in subli]
    all_ptokens =  [li for subli in puncted_tokens for li in subli]
    all_nstamps = [li for subli in timestamps for li in subli]
    rev_normalizer = ReverseNormalizer(etagger, all_ntokens, all_ptokens)


    tokens = [token for token in all_ntokens if token != '']
    tokens = [word for phrase in tokens for word in phrase.split()]

    #for ind, pind in normal_to_puncted_offsets.items():
    #  print(ind, pind, tokens[ind], all_ptokens[pind])

    timestamps = all_nstamps
    if len(tokens) != len(timestamps):
      return []

    # Filter out invalid tokens (with None timestamps)

    valid_tokens = [
        (token, start, end, i)
        for i, (token, (start, end)) in enumerate(zip(tokens, timestamps))
        if start is not None and end is not None
    ]
    filtered_tokens, start_times, end_times, indices = zip(*valid_tokens) if valid_tokens else ([], [], [], [])

    chunks = []
    total_duration = end_times[-1] if end_times else 0  # Determine total duration from the last token's end time

    # Start chunking
    start_time = 0
    #print(tokens[68], tokens[69], tokens[70])
    #print(timestamps[68], timestamps[69], timestamps[70])
    while start_time < total_duration:
        # Define chunk boundaries
        left_start = max(0, start_time - left_window_ms)
        mid_start = start_time
        mid_end = start_time + mid_window_ms
        right_end = mid_end + right_window_ms

        # Extract tokens for each section
        #print(left_start, mid_start, mid_end, right_end, "in loop times")
        left_tokens, left_timestamps, left_indices = extract_tokens_in_range(
            filtered_tokens, start_times, end_times, indices, left_start, mid_start
        )
        mid_tokens, mid_timestamps, mid_indices = extract_tokens_in_range(
            filtered_tokens, start_times, end_times, indices, mid_start, mid_end
        )

        right_tokens, right_timestamps, right_indices = extract_tokens_in_range(
            filtered_tokens, start_times, end_times, indices, mid_end, right_end
        )

        # Build JSON objects for left, mid, and right
        left_json = build_section_json(left_tokens, left_timestamps, left_indices, tokens)
        mid_json = build_section_json(mid_tokens, mid_timestamps, mid_indices, tokens)
        right_json = build_section_json(right_tokens, right_timestamps, right_indices, tokens)

        # Save the chunk
        chunks.append({
            "left": left_json,
            "mid": mid_json,
            "right": right_json,
        })

        # Move the start time forward by the mid window size
        start_time += mid_window_ms

    chunks = [chunk for chunk in chunks if chunk["mid"]["start_time"] and chunk["mid"]["end_time"]]
    newchunks = []
    for chunk in chunks:
      try:
        chunk = rev_normalizer.reverse_normalize_chunk(chunk)
        newchunks.append(chunk)
      except:
        pass
    return newchunks


def build_section_json(tokens, timestamps, indices, all_tokens):
    """
    Build a JSON object for a section (left, mid, or right) with metadata.

    Args:
        tokens (list[int]): List of tokens in the section.
        timestamps (list[tuple]): List of (start, end) timestamps for the tokens.
        indices (list[int]): List of original indices for the tokens.

    Returns:
        dict: JSON object containing tokens, start_time, end_time, start_index, and end_index.
    """
    if not tokens:  # Empty section
        return {
            "tokens": [],
            "start_time": None,
            "end_time": None,
            "start_index": None,
            "end_index": None,
        }

    start_time = timestamps[0][0]  # The start time of the first token
    end_time = timestamps[-1][1]  # The end time of the last token
    start_index = indices[0]  # The index of the first token
    end_index = indices[-1] +1  # The index of the last token
    tokens = all_tokens[start_index:end_index]

    return {
        "tokens": tokens,
        "start_time": start_time,
        "end_time": end_time,
        "start_index": start_index,
        "end_index": end_index,
    }



def chunk_directory(directory, left_window_ms=5000, mid_window_ms=20000, right_window_ms=5000):
    """
    Chunk tokens and timestamps from all files in a directory.

    Args:
        directory (str): Path to the directory containing token/timestamp files.
        left_window_ms (int): Size of the left window in milliseconds.
        mid_window_ms (int): Size of the mid section in milliseconds.
        right_window_ms (int): Size of the right window in milliseconds.

    Returns:
        list[dict]: List of chunks from all files, each containing global chunk number and filename.
    """
    global_chunk_number = 0  # Global counter for chunk numbers
    all_chunks = []  # To store chunks from all files

    # Loop through all files in the directory
    root_paths = []
    for root, _, files in os.walk(directory):
            root_paths.append((root, files))

    root_paths.sort(key=lambda x: x[0])
    root_paths = [filep for filep in root_paths if filep[0][8:] not in finished_files and filep[0][8:][0] >= 'S']

    with open("all_chunks.jsonl", "a+") as ofile:
      for root, files in root_paths:
        for filename in files:
          if filename == "aligned.swc":
            file_path = os.path.join(root, filename)
            directory_name = os.path.basename(root)
            print(directory_name)
            file_chunks = chunk_single_file(file_path, left_window_ms, mid_window_ms, right_window_ms)
            all_chunks = []

            for chunk in file_chunks:
              mid = chunk["mid"]
              if not mid["tokens"] or not mid["start_time"] or not mid["end_time"]:
                continue
              chunk["global_chunk_number"] = global_chunk_number
              chunk["filename"] = filename
              all_chunks.append(chunk)
              global_chunk_number += 1
            bigjs = {"filename": directory_name, "chunks": all_chunks}
            bigjs = json.dumps(bigjs)
            ofile.write(bigjs + "\n")



def extract_tokens_in_range(tokens, start_times, end_times, indices, time_start, time_end):
    """
    Extract tokens, timestamps, and indices that fall within a specific time range.

    Args:
        tokens (list[int]): List of token IDs.
        start_times (list[int]): List of start times (in milliseconds).
        end_times (list[int]): List of end times (in milliseconds).
        indices (list[int]): List of original token indices.
        time_start (int): Start of the time range (in milliseconds).
        time_end (int): End of the time range (in milliseconds).

    Returns:
        tuple: (tokens_in_range, timestamps_in_range, indices_in_range)
    """
    tokens_in_range = []
    timestamps_in_range = []
    indices_in_range = []

    for token, start, end, index in zip(tokens, start_times, end_times, indices):
        # Include tokens if they overlap with the time range
        if time_start <= start < time_end : # if end exceeds a bit, we tolerate
            tokens_in_range.append(token)
            timestamps_in_range.append((start, end))
            indices_in_range.append(index)

    return tokens_in_range, timestamps_in_range, indices_in_range

if __name__ == "__main__":
  #chunks = chunk_single_file("examples/dollars.xml")
  #print(chunks)
  chunk_directory("english")

