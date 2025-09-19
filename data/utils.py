INDEX_TO_KEY = [
    "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min", "C:maj", "C:min",
    "C#:maj", "C#:min", "D:maj", "D:min", "D#:maj", "D#:min", "E:maj", "E:min",
    "F:maj", "F:min", "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min"
]
KEY_TO_INDEX = {key: i for i, key in enumerate(INDEX_TO_KEY)}

FN_KEY_REPLACE = lambda x: x.replace("Ab", "G#").\
                             replace("Bb", "A#").\
                             replace("Db", "C#").\
                             replace("Cb", "B").\
                             replace("Eb", "D#").\
                             replace("Gb", "F#").\
                             replace("B#", "C")

INDEX_TO_MAJMIN_CHORD = [
    "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min", "C:maj", "C:min",
    "C#:maj", "C#:min", "D:maj", "D:min", "D#:maj", "D#:min", "E:maj", "E:min",
    "F:maj", "F:min", "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min"
]
MAJMIN_CHORD_TO_INDEX = {key: i for i, key in enumerate(INDEX_TO_MAJMIN_CHORD)}