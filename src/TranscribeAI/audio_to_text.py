import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper

def mkv_to_audio(mkv_file, audio_file='audio.wav'):
    print(f'Converting {mkv_file} to audio...')
    video_clip = VideoFileClip(mkv_file)
    video_clip.audio.write_audiofile(audio_file)
    video_clip.close()
    print('Audio extraction complete.')
    return audio_file

def transcribe_audio(audio_file, output_txt='transcript.txt', model_size='base'):
    print(f'Transcribing {audio_file} using Whisper...')
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file)
    transcript = result['text']

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f'Transcription complete, saved as {output_txt}')
    return output_txt

if __name__ == "__main__":
    mkv_file = r'2025-03-19 12-16-11.mkv'
    audio_file = 'output_audio.wav'
    transcript_file = 'transcript.txt'

    audio = mkv_to_audio(mkv_file, audio_file)
    transcribe_audio(audio, transcript_file)
