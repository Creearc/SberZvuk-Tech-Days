from audio import *
import urllib.request

def start(link,prefix):
    video_path = 'output/tmp_{}_video.mp4'.format(prefix)
    audio_path = 'output/tmp_{}_audio.wav'.format(prefix)
    urllib.request.urlretrieve(link, video_path)
    extract_audio(video_path,audio_path)
    print('audio is extracted!')
    change_num_channels(audio_path)
    print('chanels are changed!!')
    artists = names_words_collect(audio_path)
    print('names are collected!!')
    rus_words = rus_words_collect(audio_path)
    print('rus words are collected!!')
    text = final_set_collect(rus_words,artists)
    res_names = final_names_collect(text,'output/{}_audio.json'.format(prefix))
    audio_change(res_names,audio_path,audio_path)
    print('Done')



