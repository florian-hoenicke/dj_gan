import pretty_midi as pretty_midi

midi_pretty_format = pretty_midi.PrettyMIDI('../data/midis/midis/TV_Themes_www.tv-timewarp.co.uk_MIDIRip/a-team/A-Team.mid')
# for i, instrument in enumerate(midi_pretty_format.instruments):
# midi_pretty_format.instruments = [instrument]
midi_pretty_format.write(f'../data/experiment/a-team-original.mid')
midi_pretty_format.time_signature_changes = []
midi_pretty_format.write(f'../data/experiment/a-team.mid')
