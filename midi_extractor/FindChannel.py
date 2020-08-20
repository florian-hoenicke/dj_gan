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
        for message in self.midi:...
Shubham Ankit
a year ago
Working with MIDI data in Python using Mido

Originally published by SAM AGNEW at twilio.com

Let's walk through the basics of working with MIDI data using the Mido Python library.
Setting up

Before moving on, you will need to make sure you have an up to date version of Python 3 and pip installed. Make sure you create and activate a virtual environment before installing Mido.

Run the following command to install Mido in your virtual environment:

pip install mido==1.2.9

In the rest of this post, we will be working with these two MIDI files as examples. Download them and save them to the directory where you want your code to run.

VampireKillerCV1.mid is the stage 1 music from the original Castlevania game on the NES. VampireKillerCV3.mid is the Castlevania 3 version of the same song, which is slightly remixed and plays later in the game when you enter Dracula's Castle (hence the track name "Deja Vu")!

Working with MidiFile objects and track data

The MidiFile object is one of the basic types you're going to work with when using Mido. As the name implies, it can be used to read, write and play back MIDI files. You can create a new MIDI file or open up an existing file as a MidiFile object. Any changes you make to this object won't be written until you call the save() method with a filename.

Let's start by opening VampireKillerCV1.mid and examining its properties. Open up your Python shell and follow along with this code which will create a new MidiFile object with the contents of the file we want:

from mido import MidiFile

mid = MidiFile('VampireKillerCV1.mid', clip=True)
print(mid)

We are using clip=True just in case we end up opening a file with notes over 127 velocity, the maximum for a note in a MIDI file. This isn't a typical scenario and would usually mean the data is corrupted, but when working with large amounts of files it's good to keep in mind that this is a possibility. This would clip the velocity of all notes to 127 if they are higher than that.

You should see some output similar to this:

<midi file 'VampireKillerCV1.mid' type 1, 9 tracks, 4754 messages>

This means that the MIDI file has 9 synchronous tracks, with 4754 messages inside of them. Each MidiFilehas a type property that designates how the tracks interact with each other.

There are three types of MIDI files:

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others

Let's loop through some of the tracks and see what we find:

for track in mid.tracks:
    print(track)

Your output should look something like this:

<midi track '' 5 messages>
<midi track 'CV1- Vampire Killer' 7 messages>
<midi track 'Staff-2' 635 messages>
<midi track 'Staff-3' 659 messages>
<midi track 'Staff-4' 728 messages>
<midi track 'Staff-5' 635 messages>
<midi track 'Staff-6' 659 messages>
<midi track 'Staff-7' 1421 messages>
<midi track 'Staff-1' 5 messages>

This allows you to see the track titles and how many messages are in each track. You can loop through the messages in a track:

for msg in mid.tracks[0]:
    print(msg)

This particular track contains only meta information about the MIDI file in general such as the tempo and time signature, but other tracks contain actual musical data:

<meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
<meta message key_signature key='C' time=0>
<meta message smpte_offset frame_rate=24 hours=33 minutes=0 seconds=0 frames=0 sub_frames=0 time=0>
<meta message set_tempo tempo=468750 time=0>
<meta message end_of_track time=0>

Now let's actually do something with this info!
Manipulating MIDI tracks and writing new files

If you've opened this file in a music program such as GarageBand, you might have noticed that there are duplicate tracks for the main melody and harmony (corresponding to the NES square wave channels 1 and 2 in the source tune). This kind of thing is pretty common when dealing with video game music MIDIs, and might seem unnecessary.

Let's write some code to clean thi
            n = n + 1
        return n

    def save_midi(self, path="data/channel"):
        new_mid = MidiFile()
        new_mid.tracks.append(self.midi)
        new_mid.ticks_per_beat = self.tpb
        new_mid.save(filename=path)


if __name__ == "__main__":
    mid = MidiFile("data/backstreetboys.mid")
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
            channel.save_midi("data/backstreetboys_channel"+str(i))

    if mid.type == 0:
        for track in mid.tracks:
            new_track = MidiTrack()
            for c in channel:
                for msg in track:
                    prev_time = 0
                    try:
                        if msg.channel == c:
                            new_track.append(msg)
                        if msg.channel != c:
                            prev_msg = new_track[-1]
                            new_track[-1] = prev_msg.copy(time = msg.time + prev_msg.time)
                    except:
                        new_track.append(msg)
            #print(new_track)
            channel = Channel(new_track, mid.ticks_per_beat)
            channel.save_midi(path="data/backstreetboys_channel"+str(c))
