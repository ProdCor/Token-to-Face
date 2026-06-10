### Discrete Encoders Probing

For reproducting the probing results from the paper (normalized entropy and determination score), run the general_analysis.py script for the discrete encoders. 

The merge_tokens.py script is meant to merge intermediate representation from SpeechTokenizer, which has multiple RVQ layers, from more semantic-aligned (first) to more phonetic-aligned (last).

### HuBERT Encoder Probing

For the HuBERT encoder, run the scripts in the hubert_probing for the HuBERT encoder.

In order to fairly compare speech representations during probing, we transform HuBERT representations into a discrete form via k-means. The merge_hubert_parts.py script is meant to aggregate the HuBERT representations in one, since the extract_hubert_features.py script saved the continuous representations from the audio samples incrementally to avoid OOM.

