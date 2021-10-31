from moviepy.editor import *


def composit(audio_name, video_name):
    videoclip = VideoFileClip(video_name)
    audioclip = AudioFileClip(audio_name)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile("new_filename.mp4")


if __name__ == "__main__":
    composit("clean.wav", "output.mp4")
