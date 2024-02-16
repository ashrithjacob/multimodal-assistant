import whisperx
import gc
import os
from dotenv import load_dotenv
from pytube import YouTube
from pydub import AudioSegment
import moviepy.editor as mp
import re
import torch
from openai import OpenAI

load_dotenv()


class Preprocess:
    sub_folder = "./samples/"

    """
    run the following command to convert youtube to audio:
    yt-dlp -x --audio-format wav -o ./samples/test.wav https://www.youtube.com/watch?v=DcWqzZ3I2cY&t=7252s&ab_channel=LexFridman
    """

    @classmethod
    def split_audio_file(cls, audio_file, chunk_size_in_minutes, split_dir):
        output_dir = os.path.join(cls.sub_folder, split_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_extension = audio_file.split(".")[-1]
        audio_file_path = os.path.join(cls.sub_folder, audio_file)
        sound = AudioSegment.from_file(audio_file_path, format=file_extension)

        chunk_size_in_milliseconds = int(chunk_size_in_minutes * 60 * 1000)
        for i in range(0, len(sound), chunk_size_in_milliseconds):
            sound_bite = sound[i : i + chunk_size_in_milliseconds]
            time_stamp = int(i / 1000)
            with open(f"{output_dir}/audio_{time_stamp}.wav", "wb") as out_f:
                sound_bite.export(out_f, format="wav")

    def sort_filenames(filenames):
        return sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()))


class Transcribe:
    def __init__(self, device, batch_size, compute_type, HF_TOKEN):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.HF_TOKEN = HF_TOKEN

    def transcribe(self, audio_file, output_file):
        model = whisperx.load_model(
            "large", self.device, compute_type=self.compute_type
        )
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=self.batch_size)
        # print(result["segments"])  # before alignment

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        # print(result["segments"]) # after alignment

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.HF_TOKEN, device=device
        )

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        ref_time = int(re.search(r"\d+", audio_file).group())
        result = whisperx.assign_word_speakers(diarize_segments, result)
        for sentence in result["segments"]:  # segments are now assigned speaker IDs
            try:
                output_sentence = f"start:{sentence['start']+ref_time}| end:{sentence['end']+ref_time}| speaker:{sentence['speaker']}| text:{sentence['text']}"
            except Exception as e:
                print(e)
            with open(output_file, "a") as f:
                f.write(output_sentence + "\n")

    def run_on_split_audio(self, split_dir, output_file):
        filenames = Preprocess.sort_filenames(os.listdir(f"{split_dir}"))
        for file in filenames:
            stt.transcribe(os.path.join(split_dir, file), output_file=output_file)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
    device = "cuda"
    batch_size = 16
    compute_type = "float16"
    split_dir = os.path.join(Preprocess.sub_folder, "sharktank")

    """
    Preprocess.split_audio_file(
        audio_file="sharktank.wav", chunk_size_in_minutes=2, split_dir="sharktank"
    )
    """

    stt = Transcribe(device, batch_size, compute_type, HF_TOKEN)
    stt.run_on_split_audio(split_dir, output_file="./text/shark.txt")
