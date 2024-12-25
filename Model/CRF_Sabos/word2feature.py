# function to extract features from a sentence
def word2feature(seq, i):
    word, _ = seq[i]  # Extract the word from the tuple
    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        prev_word, _ = seq[i-1]  # Extract the previous word
        features.update({
            "prev_word.lower()": prev_word.lower(),
            "prev_word.isupper()": prev_word.isupper(),
            "prev_word.istitle()": prev_word.istitle(),
        })
    else:
        features["BOS"] = True  # Beginning of seq

    if i < len(seq) - 1:
        next_word, _ = seq[i+1]  # Extract the next word
        features.update({
            "next_word.lower()": next_word.lower(),
            "next_word.isupper()": next_word.isupper(),
            "next_word.istitle()": next_word.istitle(),
        })
    else:
        features["EOS"] = True  # End of seq

    return features


