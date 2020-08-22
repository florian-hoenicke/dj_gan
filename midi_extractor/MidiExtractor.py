from mido import MidiFile, MidiTrack
from glob import glob


class MidiExtractor:
    def __init__(self):
        pass

    def extract_all(self, folder):
        midis = []
        for file in glob(folder):
            midi = MidiFile(file, clip=True)


    def extract(self, file_path):
        mid = MidiFile(file_path, clip=True)
        new_track = MidiTrack()
        channel_to_message = {}
        for message in mid.tracks[0]:
            if message.type == 'note_on' or message.type == 'note_off':
                channel = message.channel

    def get_message_types(self, track):
        type_set = set()
        for message in mid.tracks[0]:
            type_set.add(message.type)
        return type_set

if __name__ == "__main__":
    mid = MidiFile('data/Batman.mid', clip=True)
    new_track = MidiTrack()
    types = MidiExtractor().get_message_types(mid.tracks[0])
    print(types)
    for message in mid.tracks[0]:
        # if message.type == 'note_on':
        #     channel =
        if message.type != 'note_on' or message.channel == 9:
             new_track.append(message)
        print(message)
        print()
    mid.tracks[0] = new_track
    # mid.ticks_per_beat = 1000
    mid.save(filename='data/higher_pitch.mid')
