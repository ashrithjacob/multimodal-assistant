from faster_whisper import WhisperModel
from dotenv import load_dotenv
import torch
import os

load_dotenv()
from pyannote.audio import Pipeline

class Audio:
    @classmethod
    def transcript_audio(cls, audio_path):
        # Run on GPU with FP16
        model_size = "large-v3"
        model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        print(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        return segments

    @classmethod
    def diaterize(cls, audio_path):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGING_FACE_API_KEY"),
        )

        # send pipeline to GPU (when available)
        pipeline.to(torch.device("cuda"))

        # apply pretrained pipeline
        diarization = pipeline(audio_path)

        # print the result
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


if __name__ == "__main__":
    audio_path = "./first_person_audio.wav"
    # use export LD_LIBRARY_PATH=/home/ash/miniconda3/envs/whisper/lib/python3.10/site-packages/torch/lib/
    #segments = Audio.transcript_audio(audio_path)
    #for segment in segments:
    #    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    #os.unsetenv('LD_LIBRARY_PATH')
    # unset LD_LIBRARY_PATH
    Audio.diaterize(audio_path)
