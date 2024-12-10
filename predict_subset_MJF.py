import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
)
from matplotlib import pyplot as plt

# Set the models directory
MODELS_HOME = Path("./models")


class Predictor:
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.classifiers = {}
        self.sample_rate = 16000
        self.chunk_size = 10  # seconds per chunk

    def setup(self):
        """Load the models into memory and set up the configurations."""
        print("[INFO] Loading models...")

        self.models = {
            "msd-musicnn": str(MODELS_HOME / "msd-musicnn-1.pb"),
            "audioset-vggish": str(MODELS_HOME / "audioset-vggish-3.pb"),
            "emomusic-msd-musicnn": str(MODELS_HOME / "emomusic-msd-musicnn-1.pb"),
            "emomusic-audioset-vggish": str(MODELS_HOME / "emomusic-audioset-vggish-1.pb"),
            "deam-msd-musicnn": str(MODELS_HOME / "deam-msd-musicnn-1.pb"),
            "deam-audioset-vggish": str(MODELS_HOME / "deam-audioset-vggish-1.pb"),
            "muse-msd-musicnn": str(MODELS_HOME / "muse-msd-musicnn-1.pb"),
            "muse-audioset-vggish": str(MODELS_HOME / "muse-audioset-vggish-1.pb"),
        }

        self.embeddings = {
            "msd-musicnn": TensorflowPredictMusiCNN(
                graphFilename=self.models["msd-musicnn"],
                output="model/dense/BiasAdd",
                patchHopSize=187,
            ),
            "audioset-vggish": TensorflowPredictVGGish(
                graphFilename=self.models["audioset-vggish"],
                output="model/vggish/embeddings",
                patchHopSize=96,
            ),
        }

        # Set up classifiers for each dataset and embedding type
        datasets = ["emomusic", "deam", "muse"]
        for dataset in datasets:
            for embedding_type in self.embeddings:
                classifier_name = f"{dataset}-{embedding_type}"
                self.classifiers[classifier_name] = TensorflowPredict2D(
                    graphFilename=self.models[f"{dataset}-{embedding_type}"],
                    input="flatten_in_input",
                    output="dense_out",
                )
        print("[INFO] Models loaded successfully!")

    def predict(self, audio: Path, folder_name: str, overall_csv: Path):
        """Run a single prediction on the model"""
        print(f"Processing file: {audio}")
        loader = MonoLoader(filename=str(audio), sampleRate=self.sample_rate)
        waveform = loader()

        embeddings = self.embeddings["msd-musicnn"](waveform)
        results = self.classifiers["emomusic-msd-musicnn"](embeddings)
        results = np.mean(results.squeeze(), axis=0)

        # Normalize (1, 9) -> (-1, 1)
        valence = (results[0] - 5) / 4
        arousal = (results[1] - 5) / 4

        # Determine emotion label
        emotion = self.get_emotion_label(valence, arousal)

        # Extract song title for graph and CSV
        song_title = self.get_graph_title(audio.name)

        # Save predictions to folder-specific CSV
        self.save_to_csv(folder_name, song_title, valence, arousal, emotion)

        # Save to overall CSV
        with open(overall_csv, "a") as f:
            f.write(f"{folder_name},{song_title},{valence:.4f},{arousal:.4f},{emotion}\n")

        # Create plot
        self.create_plot(folder_name, song_title, valence, arousal, emotion)

    def save_to_csv(self, folder_name, song_title, valence, arousal, emotion):
        """Save valence, arousal, and emotion to folder-specific CSV"""
        folder_path = Path(f"./{folder_name}")
        csv_file = folder_path / f"{folder_name}_prediction.csv"

        # Ensure folder exists
        folder_path.mkdir(parents=True, exist_ok=True)

        # Write header if CSV doesn't exist
        if not csv_file.exists():
            with open(csv_file, "w") as f:
                f.write("FolderName,SongTitle,Valence,Arousal,Emotion\n")

        # Append new prediction
        with open(csv_file, "a") as f:
            f.write(f"{folder_name},{song_title},{valence:.4f},{arousal:.4f},{emotion}\n")

    def get_emotion_label(self, valence, arousal):
        """Map valence and arousal to emotional labels"""
        if valence > 0.5 and arousal > 0.5:
            return "Excitement"
        elif valence > 0.5 and -0.25 <= arousal <= 0.5:
            return "Contentment"
        elif valence > 0.5 and arousal < -0.25:
            return "Relaxation"
        elif -0.25 <= valence <= 0.5 and arousal > 0.5:
            return "Alertness"
        elif -0.25 <= valence <= 0.5 and -0.25 <= arousal <= 0.5:
            return "Calm"
        elif -0.25 <= valence <= 0.5 and arousal < -0.25:
            return "Sadness"
        elif valence < -0.25 and arousal > 0.5:
            return "Anger"
        elif valence < -0.25 and -0.25 <= arousal <= 0.5:
            return "Melancholy"
        elif valence < -0.25 and arousal < -0.25:
            return "Fatigue"
        else:
            return "Unknown"

    def get_graph_title(self, file_name):
        """Extract title from filename between the first and second '-x-'"""
        parts = file_name.split(" -x- ")
        if len(parts) > 1:
            return parts[1].strip()
        return file_name

    def create_plot(self, folder_name, song_title, valence, arousal, emotion):
        """Generate and save a scatter plot"""
        sns.set_style("darkgrid")
        plt.figure(figsize=(6, 6))

        plt.scatter(valence, arousal, s=100)
        plt.axhline(0, color="grey", linewidth=0.8)
        plt.axvline(0, color="grey", linewidth=0.8)
        plt.title(f"{song_title} ({emotion})", fontsize=14, pad=20)
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        folder_path = Path(f"./{folder_name}")
        out_path = folder_path / f"{song_title}_plot.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def process_folders(root_folder, overall_csv):
    """Process audio files in subfolders starting with 'SONGS_' or similar patterns"""
    root = Path(root_folder)
    for folder in root.iterdir():
        if folder.is_dir():
            for subfolder in folder.iterdir():
                if subfolder.is_dir() and any(
                    subfolder.name.startswith(prefix) for prefix in ["SONGS_", "Songs_", "songs_", "SONGS-", "Songs-", "songs-"]
                ):
                    folder_name = folder.name
                    audio_files = subfolder.glob("*.mp3")
                    for audio_file in audio_files:
                        predictor.predict(audio=audio_file, folder_name=folder_name, overall_csv=overall_csv)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    root_folder = "/Users/vanessabarrera/Documents/MJF/01_Audio Samples/emotion_analyze/"
    overall_csv = Path("./overall_prediction.csv")

    # Create overall CSV header
    if not overall_csv.exists():
        with open(overall_csv, "w") as f:
            f.write("FolderName,SongTitle,Valence,Arousal,Emotion\n")

    process_folders(root_folder, overall_csv)
