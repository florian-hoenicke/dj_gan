from mido import MidiFile, MidiTrack


class MidiExtractor:
    def __init__(self):
        pass
    def extract(self):
        pass

if __name__ == "__main__":
    mid = MidiFile('data/Batman.mid', clip=True)
    new_track = MidiTrack()
    for message in mid.tracks[0]:
        if message.type != 'note_on' or message.channel == 9:
             new_track.append(message)
        print(message)
        print()
    mid.tracks[0] = new_track
    # mid.ticks_per_beat = 1000
    mid.save(filename='data/higher_pitch.mid')
