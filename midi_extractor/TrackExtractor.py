import pretty_midi as pretty_midi

midi_pretty_format = pretty_midi.PrettyMIDI('../data/backstreetboys.mid')
for i, instrument in enumerate(midi_pretty_format.instruments):
    midi_pretty_format.instruments = [instrument]
    midi_pretty_format.write(f'../data/single_track{i}.mid')
