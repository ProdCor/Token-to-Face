For the reproduction of the results presented in the paper, you will need to clone the repositories for each encoder model, as well as downloading their own pre-trained weights.

I will make it available the environment and requirements I was using for each model.

### Instructions:

1 - Run the create_audio_file_directory.py script to transform the audio samples from the dataset into the correct format.

2 - Run the extract_speech_tokens_split.py script to extract the speech tokens from the audio samples in the new format.

3 - Run the main.py script to train the decoder to generate the blendshapes vectors (.npz or .npy format). Here, you can choose between training the decoder with facial masking.
(different regions from the face), or to predict the full 52D blendshape vectors.

4 - For inference, run the inference_batch.py script.

5 - If you are training different decoders for different regions of the face, run combine_regions_batch.py to get the full facial animation.

For the evaluation metrics, you will have the scripts for MVE, LVE, FDD, Jitter, Motion Intensity and High Frequency Energy in the evaluate.py script. For Bilabial Closure Score, run the lip_closure.py script. The GRU model outputs blendshape sequence in .npy format, while the Transformer model outputs it in .npz format. The choice of format is an argument in command line to run the script.

We also make it available the code to generate voice and facial animation simmultaneoulsy from text and a reference audio. For that, run avtts_pipeline.py.