from mido import MidiFile, MidiTrack, Message

from midi_extractor import convert0to1


class Channel:

    def __init__(self, midi, tpb):
        self.midi = midi
        self.tpb = tpb

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
        new_mid.tracks.append(self.midi)
        new_mid.ticks_per_beat = self.tpb
        new_mid.save(filename=path)


if __name__ == "__main__":
    mid = MidiFile("data/Batman.mid")
    tpb = mid.ticks_per_beat
    channel = []
    for track in mid.tracks:
        for msg in track:
            try:
                if msg.channel not in channel:
                    channel.append(msg.channel)
            except:
                pass
    print(channel)

    if mid.type ==1:
        for i, track in enumerate(mid.tracks):
            print(i)
            print(track)
            new_track = MidiTrack()
            for msg in track:
                new_track.append(msg)
            channel = Channel(new_track, mid.ticks_per_beat)
            channel.save_midi("data/batman_channel"+str(i))

    if mid.type == 0:
        for track in mid.tracks:

            for c in channel:
                new_track = MidiTrack()
                for msg in track:
                    prev_time = 0
                    if hasattr(msg, "channel"):
                        if msg.channel == c:
                            if hasattr(msg, "velocity") and msg.velocity == 0:
                                new_track.append(msg.copy())
                            else:
                                new_track.append(msg)
                        if msg.channel != c:
                            #new_track.append(Message("note_off", time = msg.time))
                            prev_msg = new_track[-1]
                            new_track[-1] = prev_msg.copy(time = msg.time + prev_msg.time)
                    else:
                        new_track.append(msg)
            #print(new_track)
                print(new_track)
                channel = Channel(new_track, mid.ticks_per_beat)
                channel.save_midi(path="data/batman_channel"+str(c))
