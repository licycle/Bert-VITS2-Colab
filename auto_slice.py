import librosa
import soundfile as sf
import os
import argparse
from slicer2 import Slicer

def process_audio_files(input_dir, output_dir):
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    for audio_file in audio_files:
        audio_path = os.path.join(input_dir, audio_file)
        audio, sr = librosa.load(audio_path, sr=None, mono=False)  # Load an audio file with librosa.

        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=2000,  # Set minimum length to 2000 ms (2 seconds)
            min_interval=300,
            hop_size=10,
            max_sil_kept=500
        )
        chunks = slicer.slice(audio)

        file_number = os.path.splitext(audio_file)[0].split('_')[-1]  # Extract number from filename
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T  # Swap axes if the audio is stereo.
            output_file = os.path.join(output_dir, f'{file_number}_{i}.wav')
            sf.write(output_file, chunk, sr)  # Save sliced audio files with soundfile.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument("input_dir", type=str, help="Input directory containing .wav files")
    parser.add_argument("output_dir", type=str, help="Output directory for processed files")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_audio_files(args.input_dir, args.output_dir)
