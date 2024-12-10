# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
from cmath import polar
import pandas as pd
import numpy as np
import seaborn as sns
from cog import BasePredictor, Input, Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
)
from matplotlib import pyplot as plt

# Updated to point to the local models directory
MODELS_HOME = Path("./models")

# Path to the CSV file
CSV_FILE = Path("./predictions.csv")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.musicnn_graph = str(MODELS_HOME / "msd-musicnn-1.pb")
        self.vggish_graph = str(MODELS_HOME / "audioset-vggish-3.pb")

        # Debugging: Print model paths and existence checks
        print(f"Checking TensorFlow graph file: {self.musicnn_graph}")
        print(f"File exists: {Path(self.musicnn_graph).exists()}")
        print(f"Checking TensorFlow graph file: {self.vggish_graph}")
        print(f"File exists: {Path(self.vggish_graph).exists()}")

        self.sample_rate = 16000

        self.loader = MonoLoader()
        self.embeddings = {
            "msd-musicnn": TensorflowPredictMusiCNN(
                graphFilename=self.musicnn_graph,
                output="model/dense/BiasAdd",
                patchHopSize=187,
            ),
            "audioset-vggish": TensorflowPredictVGGish(
                graphFilename=self.vggish_graph,
                output="model/vggish/embeddings",
                patchHopSize=96,
            ),
        }

        self.input = "flatten_in_input"
        self.output = "dense_out"
        # Algorithms for specific models.
        self.classifiers = {}

        datasets = ("emomusic", "deam", "muse")
        for dataset in datasets:
            for embedding in self.embeddings.keys():
                classifier_name = f"{dataset}-{embedding}"
                graph_filename = str(MODELS_HOME / f"{classifier_name}-1.pb")
                self.classifiers[classifier_name] = TensorflowPredict2D(
                    graphFilename=graph_filename,
                    input=self.input,
                    output=self.output,
                )

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to process",
            default=None,
        ),
        embedding_type: str = Input(
            description="Embedding type to use: vggish, or musicnn",
            default="msd-musicnn",
            choices=["msd-musicnn", "audioset-vggish"],
        ),
        dataset: str = Input(
            description="Arousal/Valence training dataset",
            default="emomusic",
            choices=["emomusic", "deam", "muse"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert audio, "Specify an audio filename"

        # Load audio
        print("loading audio...")
        self.loader.configure(
            sampleRate=self.sample_rate,
            filename=str(audio),
            resampleQuality=4,
        )
        waveform = self.loader()

        embeddings = self.embeddings.get(embedding_type, None)
        if embeddings is None:
            raise ValueError(f"Invalid embedding type: {embedding_type}. Choose from {list(self.embeddings.keys())}.")
        embeddings = embeddings(waveform)

        classifier_name = f"{dataset}-{embedding_type}"
        if classifier_name not in self.classifiers:
            raise KeyError(
                f"Invalid classifier name: {classifier_name}. Available classifiers are: {list(self.classifiers.keys())}"
            )

        results = self.classifiers[classifier_name](embeddings)
        results = np.mean(results.squeeze(), axis=0)

        # Manual normalization (1, 9) -> (-1, 1)
        results = (results - 5) / 4

        valence = results[0]
        arousal = results[1]

        # Save valence and arousal to a CSV file
        self.save_to_csv(audio.name, valence, arousal)

        sns.set_style("darkgrid")
        g = sns.lmplot(
            data=pd.DataFrame({"arousal": [arousal], "valence": [valence]}),
            x="valence",
            y="arousal",
            markers="x",
            scatter_kws={"s": 100},
        )

        g.set(ylim=(-1, 1))
        g.set(xlim=(-1, 1))
        plt.plot([-1, 1], [0, 0], linewidth=1.5, color="grey")
        plt.plot([0, 0], [-1, 1], linewidth=1.5, color="grey")
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.15)
        plt.title(audio.name)

        out_path = Path("./out.png")  # Save plot in the current directory
        plt.savefig(out_path)

        print("done!")
        return out_path

    def save_to_csv(self, filename, valence, arousal):
        """Save the audio filename, valence, and arousal values to a CSV file"""
        if not CSV_FILE.exists():
            # If the CSV file doesn't exist, create it with a header
            with open(CSV_FILE, "w") as f:
                f.write("Filename,Valence,Arousal\n")

        # Append the new row to the CSV file
        with open(CSV_FILE, "a") as f:
            f.write(f"{filename},{valence:.4f},{arousal:.4f}\n")
        print(f"Saved results to {CSV_FILE}")


if __name__ == "__main__":
    # Create the Predictor instance
    predictor = Predictor()
    predictor.setup()

    # Full path to your audio file
     # audio_file = Path("/Users/vanessabarrera/Documents/MJF/01_Audio Samples/emotion_analyze/00MDHA20/Songs_00MDHA20A11BD/Courtney Pine -x- Brotherman -x- The Montreux Jazz Festival Archive -x- 2000 -x- 1 -x- 47436.mp3")
     # audio_file = Path("/Volumes/Lacie_VaB23/01_Musik/TuneFab_Dec24/08 The Seed (2.0).mp3")
    
    audio_file = Path("/Users/vanessabarrera/Documents/MJF/01_Audio Samples/emotion_analyze/03MDHA50/SONGS_03MDHA50A11BD/The Roots -x- The Seed (2.0) -x- The Montreux Jazz Festival Archive -x- 2003 -x- 14 -x- 20147.mp3")
     
    

    # Run the prediction with explicit embedding type and dataset
    print(f"Processing file: {audio_file}")
    result_path = predictor.predict(audio=audio_file, embedding_type="msd-musicnn", dataset="emomusic")

    # Print or save the result
    print(f"Result saved at: {result_path}")

    
    
  