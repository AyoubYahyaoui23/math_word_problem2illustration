from lib_text2img import get_html_from_sentence
from flair.models import SequenceTagger
import spacy


def main(sentence):

    model_path = "models/checkpoint-980"
    output_path = "output2.html"
    image_folder_path = "images"
    nlp = spacy.load("models/en_core_web_lg")
    tagger_chunk = SequenceTagger.load("flair/chunk-english")
    tagger_ner = SequenceTagger.load("flair/ner-english-ontonotes-large")

    get_html_from_sentence(sentence, tagger_chunk, tagger_ner, nlp, model_path, output_path, image_folder_path)


if __name__ == "__main__":
    sentence = "Will was drawing super heroes on a sheet of scrap paper. He drew two heroes on the front and seven heroes on the back."
    # You can now use the loaded SequenceTagger without loading it again
    main(sentence)
