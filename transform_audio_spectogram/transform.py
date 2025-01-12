import os
import wave
from pathlib import Path

import pydub
from PIL import Image
from spectogram_image_converter import SpectrogramImageConverter
from spectrogram_params import SpectrogramParams

FRAME_RATE = 44100
AUDIOS = Path(__file__).parent / 'audios'
SPECTROGRAMS = Path(__file__).parent / 'spectrograms'


class Transform:
    def to_spectrogram(self, path_to_audio: str) -> Image.Image:
        """The method converts audio file to a spectrogram, if the audio has a sampling (frame) rate
        different than 44100 which is used in the other methods the audio is resampled
        so that it matches the needed value.

        Args:
            path_to_audio (str): path to an audio file in wav format

        Returns:
            Image.Image: Image object of the corresponding spectrogram
        """
        # parameters
        spec_params = SpectrogramParams()
        exif_tags = spec_params.to_exif()
        params = SpectrogramParams.from_exif(exif_tags)

        # initializing the converter
        converter = SpectrogramImageConverter(params, "cuda")

        # loading the audio segment
        audio_segment = pydub.AudioSegment.from_wav(path_to_audio)

        # checking the frame rate
        with wave.open(path_to_audio, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            print(f"Sample rate: {sample_rate} Hz")

            # change the frame rate if needed
            if sample_rate != FRAME_RATE:
                audio_segment = audio_segment.set_frame_rate(FRAME_RATE)

            spectrogram = converter.spectrogram_image_from_audio(audio_segment)

            spectrogram.save(os.path.join(SPECTROGRAMS, Path(path_to_audio).stem + '.jpg'))
            # spectrogram.show()

            return spectrogram


    def to_audio(self, path_to_spectrogram: str) -> pydub.AudioSegment:
        """The method converts an image of a spectrogram to an audio wav file.

        Args:
            path_to_audio (str): path to an image of the spectrogram in jpg format

        Returns:
            pydub.AudioSegment: pydub.AudioSegment object of the corresponding audio wav file
        """
        # parameters
        spec_params = SpectrogramParams()
        exif_tags = spec_params.to_exif()
        params = SpectrogramParams.from_exif(exif_tags)

        # initializing the converter
        converter = SpectrogramImageConverter(params, "cuda")

        spectogram = Image.open(path_to_spectrogram)
        audio = converter.audio_from_spectrogram_image(spectogram)
        audio.export(os.path.join(AUDIOS, Path(path_to_spectrogram).stem + '.wav'), format="wav")

        return audio


# testing
if __name__ == '__main__':
    transformer = Transform()
    # transformer.to_spectrogram(os.path.join(AUDIOS, 'hiphop1.wav'))
    # transformer.to_audio(os.path.join(SPECTROGRAMS, 'clip.jpg'))
