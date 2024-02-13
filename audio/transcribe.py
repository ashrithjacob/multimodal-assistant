import whisperx
import gc
import os
from dotenv import load_dotenv
from pytube import YouTube
from pydub import AudioSegment
import moviepy.editor as mp
import re
from openai import OpenAI

load_dotenv()


def youtube_to_audio(youtube_url, output_file):  # use command in doc.md instead
    try:
        # Download YouTube video
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(output_path="./", filename=output_file)
    except Exception as e:
        print("Error:", e)


def sort_filenames(filenames):
    return sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()))


def split_mp3(audio_file, chunk_size_in_minutes=0.1, output_dir="./samples/jeff"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_extension = audio_file.split(".")[-1]
    sound = AudioSegment.from_file(audio_file, format="mp3")
    chunk_size_in_milliseconds = int(chunk_size_in_minutes * 60 * 1000)
    for i in range(0, len(sound), chunk_size_in_milliseconds):
        sound_bite = sound[i : i + chunk_size_in_milliseconds]
        with open(f"{output_dir}/audio_{i}.wav", "wb") as out_f:
            sound_bite.export(out_f, format="wav")


def query_chatbot(question):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open("speech.txt", "r") as file:
        chat = file.read()

    chat_and_question = f"Chat:{chat}\n{question}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers querstions based on the given chat",
            },
            {"role": "user", "content": chat_and_question},
        ],
    )
    print(response.choices[0].message.content)


class Transcribe:
    def __init__(self, device, batch_size, compute_type, HF_TOKEN):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.HF_TOKEN = HF_TOKEN

    def transcribe(self, audio_file):
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

        ref_time = int(re.search(r"\d+", audio_file).group()) / 1000
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        print("SEGMENTS AFTER DIATERIZATION:")
        for sentence in result["segments"]:  # segments are now assigned speaker IDs
            output_sentence = f"start:{sentence['start']+ref_time}| end:{sentence['end']+ref_time}| speaker:{sentence['speaker']}| text:{sentence['text']}"
            with open("speech.txt", "a") as f:
                f.write(output_sentence + "\n")


if __name__ == "__main__":
    HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
    device = "cuda"
    batch_size = 16
    compute_type = "float16"
    audio_folder = "./samples/jeff"

    """
    query_chatbot("Provide me an exhaustive list of occiurences in which John Glenn was mentioned?")
    """

    # takes in a file number and transcribes the audio file and adds the transcript to the speech.txt file
    """
    file_number = 5
    filenames = sort_filenames(os.listdir(f"{audio_folder}"))
    print(filenames)
    stt = Transcribe(device, batch_size, compute_type, HF_TOKEN)

    # set filenames to the audio
    file = filenames[file_number]
    stt.transcribe(f"{audio_folder}/{file}")
    """

    # split mp3 into chunks
    """ 
    output_file = "./samples/jeff.mp3"
    split_folder = "./samples/jeff"
    split_mp3(output_file, chunk_size_in_minutes=2, output_dir=split_folder)
    """
# pytube audio doesnt split with pydub splitter. try youtube-dl
