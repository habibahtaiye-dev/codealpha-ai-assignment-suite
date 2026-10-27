"""Educational LSTM music-generation demo using synthetic note patterns.

This creates toy training sequences so the example can run without downloading
copyrighted MIDI data. For a real project, train on MIDI files you are licensed
to use and document the dataset source.
"""
import numpy as np

def make_sequences(count=500, length=16):
    # Toy note sequences in a C-major-like range; this is not real music data.
    scale = np.array([60, 62, 64, 65, 67, 69, 71, 72])
    data = []
    rng = np.random.default_rng(7)
    for _ in range(count):
        start = int(rng.integers(0, len(scale)))
        seq = [int(scale[start])]
        for _ in range(length - 1):
            step = int(rng.choice([-1, 0, 1], p=[0.3, 0.2, 0.5]))
            start = (start + step) % len(scale)
            seq.append(int(scale[start]))
        data.append(seq)
    return np.asarray(data, dtype=np.int32)

def main():
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Embedding
        from music21 import stream, note, tempo
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install tensorflow music21") from exc

    sequences = make_sequences()
    x = sequences[:, :-1]
    y = sequences[:, -1]
    model = Sequential([
        Embedding(input_dim=128, output_dim=16, input_length=x.shape[1]),
        LSTM(48),
        Dense(128, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=3, batch_size=32, verbose=1)

    seed = list(sequences[0, :-1])
    generated = seed[:]
    for _ in range(32):
        window = np.array(generated[-(x.shape[1]):], dtype=np.int32)[None, :]
        probs = model.predict(window, verbose=0)[0]
        # Restrict sampling to notes in the toy scale.
        allowed = np.array([60, 62, 64, 65, 67, 69, 71, 72])
        candidate_probs = probs[allowed]
        candidate_probs = candidate_probs / candidate_probs.sum() if candidate_probs.sum() else np.ones(len(allowed))/len(allowed)
        generated.append(int(np.random.choice(allowed, p=candidate_probs)))

    score = stream.Stream()
    score.append(tempo.MetronomeMark(number=100))
    for pitch in generated:
        score.append(note.Note(pitch, quarterLength=0.5))
    score.write("midi", fp="generated_notes.mid")
    print("Saved generated_notes.mid. Note: this is a toy educational output.")

if __name__ == "__main__":
    main()
