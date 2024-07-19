"""
On Q2ID
In summary, this script is processing meeting transcripts, optionally chunking them, 
and evaluating their similarity to a target answer using the ROUGE-1 f1 score metric. 
The results are written to an output file for further analysis.
"""
import os
import sys
import json

from tqdm import tqdm
from datasets import load_metric

sys.setrecursionlimit(10000)
metric = load_metric("rouge")

if __name__ == "__main__":
    do_chunks = int(sys.argv[1])
    if do_chunks:
        from transformers import AutoTokenizer
        from dataset import ChunkTokenizer
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
        # bart large input limit is 1024

        chunk_size = 256
        max_num_chunks = 128
        pad = False
        stride = True
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            stride=stride,
            pad=pad
        )
    
    # id to meeting transcripts for 3 splits
    for split in ["train", "val", "test"]:
        # id2meetingsrc = {}
        # fname = os.path.join(os.path.dirname( __file__ ), "..", "data", f'{split}-meetings.jsonl')
        # with open(fname) as f:
        #     for line in f:
        #         meeting_data = json.loads(line)
        #         if do_chunks:
        #             utts_joined = " ".join(meeting_data['meeting_transcripts'])
        #             output = chunk_tokenizer(
        #                 source=utts_joined,
        #             )
        #             input_ids = output['input_ids']
        #             tokens = tokenizer.batch_decode(input_ids)
        #             chunks = [x.replace("<pad>", "").replace("<s>", "").strip() for x in tokens]
        #             id2meetingsrc[meeting_data['meeting_id']] = chunks
        #         else:
        #             id2meetingsrc[meeting_data['meeting_id']] = meeting_data['meeting_transcripts']

        fname = os.path.join(os.path.dirname( __file__ ), "..", "data", f"{split}.jsonl")
        # with open(fname) as f:
        #     for line in f:
        #         data = json.loads(line)
        #         id2meetingsrc[data['sample_id']] = data['pos']

        if do_chunks:
            fname_out = os.path.join(os.path.dirname( __file__ ), \
                "..", "data", f"{split}.rouge.256.jsonl")
        else:
            fname_out = os.path.join(os.path.dirname( __file__ ), \
                "..", "data", f"{split}.rouge.jsonl")
        
        totals = {"train": 5000, "val": 100, "test": 258}
        with open(fname) as f, open(fname_out, "w") as out:
            for line in tqdm(f, total=totals[split]):
                data = json.loads(line)
                # meeting_utterances_final = id2meetingsrc[data['meeting_id']]
                meeting_utterances_final = data['pos']
                target = data['intent']
                query = data['query']
                # compare answer with each meeting utterance
                references = [target] * len(meeting_utterances_final)

                metric.add_batch(predictions=meeting_utterances_final, references=references)
                score = metric.compute(use_agregator=False)

                rouge_1 = score['rouge1']
                # f1 score
                scores = [x.fmeasure for x in rouge_1]
                if do_chunks:
                    data["chunks"] = meeting_utterances_final
                # key: utt_rouge_f1
                data["utt_rouge_f1"] = scores

                json.dump(data, out)
                out.write("\n")
