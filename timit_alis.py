'''
Created on Sep 15, 2014

@author: chorows
'''

import os
import glob
import cPickle as pickle

sample_rate = 16000  # samples/sec
window_length = 25e-3  # sec
window_pitch = 10e-3  # sec


def get_ali_dict(tloc):
    trans = []
    for tset in ['TRAIN', 'TEST']:
        trans.extend(glob.glob(tloc + '/' + tset + '/*/*/*.PHN'))
    alis = {}
    print len(trans)
    for i, transcription_file in enumerate(trans):
        print i
        file_parts = transcription_file.split('/')
        speaker = file_parts[-2]
        utt = file_parts[-1][:-4]
        utt_id = speaker + '_' + utt
        with open(transcription_file, 'r') as f:
            transcription = []
            for line in f:
                start, end, phone = line.strip().split()
                start = int(start)
                start = ((start - sample_rate * window_length / 2.0) /
                         (sample_rate / window_pitch))
                end = int(end)
                end = ((end - sample_rate * window_length / 2.0) /
                       (sample_rate / window_pitch))
                transcription.append((start, end, phone))
        assert utt_id not in alis
        alis[utt_id] = transcription
    return alis

path = "/data/lisa/data/timit/raw/TIMIT"
alis = get_ali_dict(path)
import ipdb; ipdb.set_trace()
