import math

import pretty_midi
from pretty_midi.instrument import Instrument
from pretty_midi.containers import Note
from midi2audio import FluidSynth

import midi2audio

class MIDITools:
    @classmethod
    def save_sequence_as_midi(cls, sequence, filename):
        test_midi = pretty_midi.PrettyMIDI('data/midis/midis/Big_Data_Set/0/040_-_Ibiza_Dreams__J.D._20110615094526.mid')
        # '/Users/florianhonicke/open_source/dj_gan/data/midis/midis/Big_Data_Set/0/040_-_Ibiza_Dreams__J.D._20110615094526.mid'
        midi = pretty_midi.PrettyMIDI(initial_tempo=80)
        notes = []
        current_time = 0
        for e in sequence:
            duration = e['durations_in']
            start = current_time
            end = current_time + duration
            current_time += duration
            velocity = max(min(e['velocity_in'], 127),0)
            tone = e['tones_in']
            octave = e['octaves_in']
            pitch = min(octave * 12 + tone, 127)
            note = Note(velocity, pitch, start, end)
            notes.append(note)
        instrument = Instrument(
            program=sequence[0]['instrument_type'],
            is_drum=sequence[0]['is_drum'],
        )
        instrument.notes = notes
        midi.instruments = [
            instrument
        ]
        midi.write(f'{filename}.midi')
        # FluidSynth().midi_to_audio(f'{filename}.midi', f'{filename}.wav')
