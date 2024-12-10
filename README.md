
# Music Emotion Recognition Scripts

This repository contains modified scripts based on the original repository from [MTG](https://github.com/MTG/essentia-replicate-demos/tree/main/music-arousal-valence). These scripts are adapted to predict the emotional content of music using valence and arousal values, with additional functionality to handle subsets of music archives such as the Montreux Jazz Festival (MJF) archive.

## Overview

The provided scripts utilize pre-trained models (MusiCNN and VGGish) to predict valence (positivity/negativity) and arousal (emotional intensity) values for audio files. The predictions can be used for tasks such as:
- Static emotional labeling (overall emotion prediction for an entire song)
- Dynamic emotional labeling (time-stamped emotional predictions for song chunks)

## Folder Structure

```plaintext
music-emotion-recognition/
├── models/
│   ├── msd-musicnn/
│   ├── audioset-vggish/
├── predict_subset_MJF.py
├── predict_onesong_dynamic.py
├── predict_onesong.py
├── README.md
```

## Requirements
- Python 3.8 or later
- TensorFlow 2.x
- Essentia
- NumPy
- Pandas
- Matplotlib

## Script Description 
1. Static Emotional Labeling for a Subset
   Script: predict_subset_MJF.py

   This script assigns a single valence, arousal, and emotional label to an entire audio file by averaging predictions across its duration.

    Key Features:
   - Loads audio files from a specified directory structure.
   - Extracts embeddings using MusiCNN or VGGish.
   - Predicts valence and arousal using pre-trained classifiers.
   - Maps predictions to emotional labels based on thresholds.
   - Outputs:
     -  CSV files with predictions for each subfolder and an aggregated file for all results.
     -  Scatter plots showing valence and arousal metrics for each song.

2. Dynamic Emotional Labeling for a Single Song
   Script: predict_onesong_dynamic.py

   Processes a single song by dividing it into fixed-length chunks (e.g., 10 seconds each). Valence and arousal are computed for each chunk, providing a temporal timeline of emotional 
   changes.

    Key Features:

   - Divides audio into 10-second segments.
   - Predicts chunk-wise valence and arousal.
   - Outputs:
     - A CSV file with start and end times, valence, and arousal for each chunk.
       
3. Studio Recordings vs. Live Concert Recordings
   Script: Compare results using both predict_onesong.py and predict_onesong_dynamic.py.

   This analysis examines the differences in emotional predictions between studio and live recordings of the same song.

   Key Features:

   - Processes studio and live recordings separately.
   - Compares static and dynamic results to evaluate acoustic variability, performance dynamics, and the impact of signal quality.

## Usage Instructions
Step 1: Clone the repository to your local machine:

```bash
git clone https://github.com/MTG/essentia-replicate-demos/tree/main/music-arousal-valence
```
Step 2: Prepare the Directory
```plaintext
music-emotion-recognition/
├── cog.yaml
├── deploy.sh
├── models/
│   ├── msd-musicnn/
│   ├── audioset-vggish/
├── predict_subset_MJF.py
├── predict_onesong_dynamic.py
├── predict_onesong.py
├── README.md
```
  Place your TensorFlow models in the models/ folder.

Step 3: Adapt the Paths in the Scripts
Before running the scripts, update the paths for your specific directory structure. For example:

- In predict_subset_MJF.py, update the root_folder variable to point to your audio folder.
- In predict_onesong_dynamic.py and predict_onesong.py, specify the audio_file and output_file paths.

Step 4: Run Scripts
## Acknowledgments
The scripts are adapted from the MTG (Music Technology Group) repository on [music emotion recognition](https://github.com/MTG/essentia-replicate-demos)

## License
These models are part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/) and are publicly available under [CC by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/) and commercial license.
