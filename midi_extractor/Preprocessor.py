import pretty_midi
import os


def midi2array(notes_list: pretty_midi.Note, instrument: int):
    """
    Converts a pretty_midi.Note object into a list of the form:
        [[pitch1, duration1, instrument1], [pitch2, duration2, instrument1],...]
    :param notes_list: pretty_midi.Noe obejct containing information about the notes of an instrument
    :param instrument: integeer to indicate the instrument referenced in notes_list
    :return: list of of notes and duration played by instrument
    """
    notes_list_formatted = []
    for note in  notes_list:
        notes_list_formatted.append([note.pitch, note.duration, instrument])
    return notes_list_formatted

class Preprocessor:
    """
    Preprocessor class to search for data and convert data to array for the GAN
    """
    def __init__(self, path_to_data):
        """

        :param path_to_data: path to data directory
        """
        self.path_to_data = path_to_data
        self.midi_list = []

    def search_midi(self):
        """
        loops through all files provided by path_to_data and matches files with extension ".mid"
        :return: generator to loop through all files
        """
        for dirpath, dirnames, filenames in os.walk(self.path_to_data):
            for filename in [f for f in filenames if f.endswith(".mid")]:
                yield os.path.join(dirpath, filename)

    def data2array(self):
        """
        loops through all midi files and all instrument channels in each file,
        generates an array for each instrument
        :return: generator to loop through all arrays of notes
        """
        i = 0
        for file in self.search_midi():
            midi = pretty_midi.PrettyMIDI(file)
            for instrument in midi.instruments:
                yield midi2array(instrument.notes, instrument.program)


if __name__ == "__main__":
    pp = Preprocessor("../data")
    i = 0
    for arr in pp.data2array():
        i += 1
        #print(arr)
    print(i)