from moviepy.editor import *


def composit(audio_name, video_name, output_name):
    videoclip = VideoFileClip(video_name)
    audioclip = AudioFileClip(audio_name)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile(output_name)


if __name__ == "__main__":
    composit("clean.wav", "output.mp4", 'result.mp4')
