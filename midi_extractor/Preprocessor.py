import pretty_midi
import os
import glob


class Preprocessor:

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.midi_list = []

    def search_midi(self):
        for dirpath, dirnames, filenames in os.walk(self.path_to_data):
            for filename in [f for f in filenames if f.endswith(".mid")]:
                yield os.path.join(dirpath, filename)

    def

if __name__ == "__main__":
    pp = Preprocessor("../data")
    pp.search_midi()
    print(pp.midi_list)