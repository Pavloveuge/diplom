#src code: https://www.kaggle.com/code/tuannguyenvananh/build-librispeech-manifest-nemo

import os
import librosa
import json

def build_manifest(subset_paths, subset):
    for subset_path in subset_paths:
        for root, dirs, files in os.walk(subset_path):
            for file in files:
                if file.endswith('.txt'):
                    transcript_path = os.path.join(root, file)
                    with open(transcript_path, 'r') as fin:
                        with open(subset + '-manifest.json', 'a') as fout:
                            for line in fin.readlines():
                                audio_id, transcript = line.split(' ', 1)
                                audio_filepath = os.path.join(root, audio_id + '.flac')
                                duration = librosa.core.get_duration(filename=audio_filepath)

                                metadata = {
                                    "audio_filepath": audio_filepath,
                                    "duration": duration,
                                    "text": transcript
                                }
                                json.dump(metadata, fout)
                                fout.write('\n')
                            
build_manifest([
        '/diplom/LibriSpeech/train-clean-100',
    ], 'train-100')