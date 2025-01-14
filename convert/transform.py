import sys
import wave

import pydub
from PIL import Image
from convert.spectogram_image_converter import SpectrogramImageConverter
from convert.spectrogram_params import SpectrogramParams

FRAME_RATE = 44100

class Transform:
    def to_spectrogram(self, path_to_audio: str, to_path: str, verbose: bool = True) -> Image.Image:
        """The method converts audio file to a spectrogram, if the audio has a sampling (frame) rate
        different from 44100 which is used in the other methods the audio is resampled
        so that it matches the needed value.

        Args:
            path_to_audio (str): path to an audio file in wav format
            verbose (bool, optional): verbosity setting

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
            
            if verbose:
                print(f"Sample rate: {sample_rate} Hz")

            # change the frame rate if needed
            if sample_rate != FRAME_RATE:
                audio_segment = audio_segment.set_frame_rate(FRAME_RATE)

            spectrogram = converter.spectrogram_image_from_audio(audio_segment)

            spectrogram.save(to_path)
            # spectrogram.show()

            return spectrogram


    def to_audio(self, path_to_spectrogram: str, to_path: str) -> pydub.AudioSegment:
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
        audio.export(to_path, format=to_path.split('.')[-1])

        return audio


# testing
if __name__ == '__main__':
    args = sys.argv
    operation = args[1]
    from_path = args[2]
    to_path = args[3]

    transformer = Transform()
    if operation == 'image2audio':
        transformer.to_audio(from_path, to_path)
    elif operation == 'audio2image':
        transformer.to_spectrogram(from_path, to_path)
    else:
        raise ValueError("Invalid operation")

