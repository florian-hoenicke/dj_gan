from mido import MidiFile, MidiTrack


class Channel:

    def __init__(self, midi):
        self.midi = midi

    def get_tone_range(self):
        min_tone = 130
        max_tone = -1
        for message in self.midi:
            if message.type == "note on" and message.note > max_tone:
                max_tone = message.note
            if message.type == "note on" and message.note < min_tone:
                min_tone = message.note
        return max_tone - min_tone

    def get_messages_count(self):
        n = 0
        for message in self.midi:
            n = n + 1
        return n

    def save_midi(self, path="data/channel"):
        new_mid = MidiFile()
        new_mid.tracks.append(new_track)
        new_mid.save(filename=path)


if __name__ == "__main__":
    mid = MidiFile('data/Batman.mid', clip=True)
    channels = []
    for message in mid.tracks[0]:
        try:
            if message.channel not in channels:
                channels.append(message.channel)
        except:
            pass
    print(channels)

    for c in channels:
        new_track = MidiTrack()
        for message in mid.tracks[0]:
            try:
                if message.channel == c:
                    new_track.append(message)
            except:
                new_track.append(message)
        channel = Channel(new_track)
        channel.save_midi(path="data/channel"+str(c))
