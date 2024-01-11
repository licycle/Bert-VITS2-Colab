import whisper
import os
import argparse
import torch

def transcribe_one(audio_path, model, lang2token, speaker):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"Detected language: {lang}")
    
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    return lang, result.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing input audio files")
    parser.add_argument("--output_file", required=True, help="Output file path for the transcriptions")
    parser.add_argument("--speaker", required=True, help="Name of the speaker")
    parser.add_argument("--languages", default="ZH", help="Language codes, default is Chinese (ZH)")
    parser.add_argument("--whisper_size", default="medium", help="Size of the Whisper model, default is 'large'")
    args = parser.parse_args()

    # Language tokens
    lang2token = {
        'zh': "ZH",
    }

    assert torch.cuda.is_available(), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    input_dir = args.input_dir
    output_file = args.output_file
    speaker = args.speaker
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(input_dir)])

    for i, wavfile in enumerate(list(os.walk(input_dir))[0][2]):
        try:
            lang, text = transcribe_one(os.path.join(input_dir, wavfile), model, lang2token, speaker)
            if lang not in list(lang2token.keys()):
                print(f"{lang} not supported, ignoring")
                continue
            formatted_text = f"{wavfile}|{speaker}|{lang2token[lang]}|{text}\n"
            speaker_annos.append(formatted_text)
            print(f"Processed: {i+1}/{total_files}")
            print(formatted_text)
        except Exception as e:
            print(e)
            continue

    if len(speaker_annos) == 0:
        print("Warning: no suitable audio files found. Please check your file structure or make sure your audio language is supported.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
