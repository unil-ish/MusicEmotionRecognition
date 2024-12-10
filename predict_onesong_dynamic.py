import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
)

class DynamicMusicEmotionPredictor:
    def __init__(self):
        self.models = {
            "audioset-vggish": "models/audioset-vggish-3.pb",
            "msd-musicnn": "models/msd-musicnn-1.pb",
        }
        self.predictors = {}
        self.chunk_size = 10  # seconds per chunk
        self.sample_rate = 44100  # Hz
        self.load_models()

    def load_models(self):
        print("[INFO] Loading TensorFlow models...")
        try:
            self.predictors["audioset-vggish"] = TensorflowPredictVGGish(
                graphFilename=self.models["audioset-vggish"],
                input="model/Placeholder",
                output="model/vggish/embeddings",
            )
            self.predictors["msd-musicnn"] = TensorflowPredictMusiCNN(
                graphFilename=self.models["msd-musicnn"]
            )
            print("[INFO] Models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            raise

    def process_audio(self, audio_path):
        loader = MonoLoader(filename=str(audio_path))
        audio = loader()
        duration = len(audio) / self.sample_rate
        print(f"Total audio duration: {duration:.2f}s")

        timestamps = []
        valence_arousal = []

        for i in range(0, len(audio), self.chunk_size * self.sample_rate):
            chunk = audio[i:i + self.chunk_size * self.sample_rate]
            start_time = i / self.sample_rate
            end_time = (i + len(chunk)) / self.sample_rate
            timestamps.append((start_time, end_time))

            if len(chunk) < self.chunk_size * self.sample_rate:
                print(f"[INFO] Skipping last chunk due to insufficient data: {start_time:.2f}s to {end_time:.2f}s")
                break

            valence, arousal = self.extract_emotion(chunk)
            valence_arousal.append((valence, arousal))
        
        return timestamps, valence_arousal

    def extract_emotion(self, audio_chunk):
        # Placeholder for emotion prediction logic
        # Replace this with your actual computation for valence and arousal
        try:
            vggish_result = self.predictors["audioset-vggish"](audio_chunk)
            musicnn_result = self.predictors["msd-musicnn"](audio_chunk)

            # Example aggregation of valence and arousal (customize as needed)
            valence = np.mean(vggish_result)  # Replace with actual valence calculation
            arousal = np.mean(musicnn_result)  # Replace with actual arousal calculation

            return valence, arousal
        except Exception as e:
            print(f"[ERROR] Error predicting emotion: {e}")
            return None, None

    @staticmethod
    def save_to_csv(timestamps, valence_arousal, output_file):
        # Combine timestamps and valence-arousal data into a DataFrame
        data = []
        for (start, end), (valence, arousal) in zip(timestamps, valence_arousal):
            data.append({"Start Time": start, "End Time": end, "Valence": valence, "Arousal": arousal})
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"[INFO] Saved results to {output_file}")


def main():
   
    audio_file = Path("/Volumes/Lacie_VaB23/01_Musik/TuneFab_Dec24/08 The Seed (2.0).mp3")
    
    output_file = "valence_arousal_predictions.csv"

    predictor = DynamicMusicEmotionPredictor()
    timestamps, valence_arousal = predictor.process_audio(audio_file)

    print("Extracted valence and arousal values for each chunk:")
    for (start, end), (valence, arousal) in zip(timestamps, valence_arousal):
        print(f"Chunk {start:.2f}s to {end:.2f}s -> Valence: {valence:.3f}, Arousal: {arousal:.3f}")

    # Save to CSV
    predictor.save_to_csv(timestamps, valence_arousal, output_file)


if __name__ == "__main__":
    main()
