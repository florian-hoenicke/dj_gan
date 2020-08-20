from collections import defaultdict
import mido, os

def subdivide_midi_tracks(path):
  '''Convert a type 0 midi file to a type 1 midi file'''
  m = mido.MidiFile(path) # load the original type-0 midi file
  messages = [] # a list of message dicts, one per track
  for track in m.tracks:
    time = 0 # store time in aggregate units
    track_messages = []
    for idx, i in enumerate(track):
      i = i.dict()
      if i.get('time', None): time += i.get('time')
      i['time'] = time
      track_messages.append(i)
    messages.append(track_messages)
  # build a dictionary of the events for each channel
  d = defaultdict(list) # d[channel_id] = [notes]
  for track_idx, track in enumerate(messages):
    for i in track:
      channel = i.get('channel', -1)
      d[channel].append(i)
  # covert time units in each program back to relative units
  for channel in d:
    total_time = 0
    for i in sorted(d[channel], key=lambda i: i['time']):
      t = i['time']
      i['time'] = t - total_time
      total_time = t
  # create a midi file and add a track for each channel
  m2 = mido.MidiFile()
  for channel in sorted(d.keys()):
    track = mido.midifiles.tracks.MidiTrack()
    # add the notes to this track
    for note in d[channel]:
      note_type = note['type']
      del note['type']
      # if this is a meta message, append a meta message else a messaege
      try:
        track.append(mido.MetaMessage(note_type, **note))
      except:
        track.append(mido.Message(note_type, **note))
    m2.tracks.append(track)
  # ensure the time quantization is the same in the new midi file
  m2.ticks_per_beat = m.ticks_per_beat
  return m2

m2 = subdivide_midi_tracks("data/Batman.mid")
print(m2)
m2.save('whoop.mid')
