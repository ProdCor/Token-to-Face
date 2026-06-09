For each model, follow the instructions:

1 - Run the create_audio_file_directory.py script to transform the audio samples from the dataset into the correct format.
2 - Run the extract_speech_tokens_split.py script to extract the speech tokens from the audio samples in the new format.
3 - Run the main.py script to train the decoder to generate the blendshapes vectors (.npz or .npy format). Here, you can choose between training the decoder with facial masking.
(different regions from the face), or to predict the full 52D blendshape vectors.
4 - For inference, run the inference_batch.py script.
5 - If you are training different decoders for different regions of the face, run combine_regions_batch.py to get the full facial animation.

We also make it available the code to generate voice and facial animation simmultaneoulsy from text and a reference audio. For that, run avtts_pipeline.py.