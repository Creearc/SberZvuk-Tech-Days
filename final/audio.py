import moviepy.editor as mp
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import json
from scipy.io import wavfile
import numpy as np
from vosk_dict import names_set
from stop_words import names_arr


def extract_audio(filepath,out):
    clip = mp.VideoFileClip(filepath)
    clip.audio.write_audiofile(out)

def change_num_channels(filepath="output.wav"):
    sound = AudioSegment.from_wav(filepath)
    sound = sound.set_channels(1)
    sound.export(filepath, format="wav")


def clear_names(names):
    names_clear =[]
    for elem in names:
        for i in range(len(elem['result'])):
            if elem['result'][i]['word'] != '[unk]':
                names_clear.append(elem['result'][i])
    return(names_clear)

def names_words_collect(filepath,modelpath='models/vosk-model-small-en-us-0.15',words=names_set):
    wf = wave.open(filepath, "rb")
    model = Model(modelpath)
    rec = KaldiRecognizer(model, wf.getframerate(), words)
    rec.SetWords(True)
    words_dict = []
    while True:
        data = wf.readframes(8000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            words_dict.append(json.loads(rec.Result()))
    clean_names = clear_names(words_dict)
    return(clean_names)

def rus_words_collect(filepath,modelpath='models/vosk-model-small-ru-0.22'):
    wf = wave.open(filepath, "rb")
    model = Model(modelpath)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    words_dict = []
    while True:
        data = wf.readframes(8000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            words_dict.append(json.loads(rec.Result()))
    return(words_dict)



def final_set_collect(rus,names):
    final_word_set = []
    for elem in rus:
        if ('result' in elem.keys()):
            for elem1 in elem['result']:
                for eng in names:
                    if  eng['end'] <= elem1['end'] and eng['conf'] >= elem1['conf'] and elem1['end'] > eng['start'] >= elem1['start'] or eng['end'] >= elem1['end'] and eng['conf'] >= elem1['conf'] and elem1['end'] > eng['start'] >= elem1['start'] or elem1['start'] < eng['end'] <= elem1['end'] and eng['conf'] >= elem1['conf'] and eng['start'] <= elem1['start'] or eng['end'] >= elem1['end'] and eng['conf'] >= elem1['conf'] and eng['start'] <= elem1['start']:
                        final_word_set.append(eng)
                        break
                    else:
                        if eng == names[-1]:
                            final_word_set.append(elem1)
    return(final_word_set)

def final_names_collect(fws,json_file_path,stop = names_arr):
    res = []
    for elem in fws:
        if elem['word'] in stop:
            res.append({"time_start": elem['start'],
                        "time_end": elem['end']})
            res_dict ={'result':res}
            with open(json_file_path, 'w') as jf:
                json.dump(res_dict,jf,indent=3)
    return(res)

def audio_change(res,filepath="output1.wav",output_filepath="clean.wav"):
    fs, data = wavfile.read(filepath)
    wav = np.array(data)
    for elem in res:
        wav[int(elem['time_start']*fs):int(elem['time_end']*fs)] = 0
    wavfile.write(output_filepath, fs, wav)




    

