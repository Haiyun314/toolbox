import string

def preprocess(file: str):
    try:
        with open(file, mode='r') as my_file:
            lines = my_file.readlines()

        word_list = []
        for sentence in lines:
            words = sentence.split()
            for word in words:
                word = word.strip(string.punctuation)  # Removes punctuation
                word_list.append(word.lower())

        print('before:', len(word_list))  # Number of words before removing duplicates
        unique_words = set(word_list)     # Remove duplicates using set
        print('after:', len(unique_words)) # Number of unique words
        return unique_words
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


def find_missing(target: list, words_box):
    record = []
    for i in target:
        if i.lower() in words_box:
            record.append(i)
    return record