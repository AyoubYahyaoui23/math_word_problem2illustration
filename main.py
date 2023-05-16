from spacy.glossary import GLOSSARY as tag_glossary
from spacy.matcher import Matcher


def extract_compound_span(doc):
    compound_list = []
    start_i = 0
    end_i = 0
    for token in doc:
        if len(compound_list) == 0 or token.i >= compound_list[-1][-1]:
            if token.dep_ == "compound" and not token.like_num:
                start_i = token.i
                if token.head.dep_ == "compound":
                    end_i = token.head.head.i + 1
                else:
                    end_i = token.head.i + 1
            if start_i != 0 and end_i != 0 and tuple([0, start_i, end_i]) not in compound_list:
                compound_list.append((0, start_i, end_i))
    return compound_list


def merge_chunks(doc, nlp):
    """
    Merge chunks of a sentence
    :param nlp:
    :param doc: spacy doc
    :return: spacy doc
    """
    matcher = Matcher(nlp.vocab)
    pattern = [
        [{"LIKE_NUM": True, "POS": "NUM"}, {"IS_PUNCT": True}, {"LIKE_NUM": True, "POS": "NUM"}],
        [{"LIKE_NUM": True, "POS": "NUM"}, {"LIKE_NUM": True, "POS": "NUM"}]]

    matcher.add("CONSECUTIVE_NUMBERS", pattern)

    matches_compound_span: list[tuple[int, int, int]] = extract_compound_span(doc)

    matches = matcher(doc)
    matches = matches_compound_span + matches
    sorted_matches = sorted(matches, key=lambda x: x[1])
    diff = 0
    for match_id, start, end in sorted_matches:
        start = start - diff
        end = end - diff
        span = doc[start:end]

        with doc.retokenize() as retokenizer:
            retokenizer.merge(span, attrs={"LEMMA": span.text})

        diff += end - start - 1

    return doc



def tag_definiton(tag):
    if tag in tag_glossary:
        return tag_glossary[tag]
    else:
        return tag


def token2features(token, sentence_id, num_value=None):
    """
    Extract features from a token
    :param token: token to extract features from
    :param sentence_id: id of the sentence in which the token is found
    :param num_value: if the token is a number, this is the value of the number
    :return: dict of features
    """
    features = {"sentence_id": sentence_id,
                "token": token.text,
                "num_value": num_value,
                "start_index": token.idx,
                "end_index": token.idx + len(token.text),
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "ner": token.ent_type_}
                # "vector": token.vector,
                # "vector_norm": token.vector_norm,
                # "tensor": token.tensor}
    # if features["ner"] == "PERSON":
    #     features["ner"] = classify_gender(features["token"])

    return features


# @tictoc
def sentence2token_features(sentence, sentence_id):
    """
    Extract tokens and their features from a sentence
    :param sentence: spacy doc object to extract features from
    :param sentence_id: id of the sentence
    :return: list of dicts of features of tokens in the sentence and tokens in the sentence
    """
    tokens = [token for token in sentence if not token.is_stop and not token.is_punct and not token.is_space]

    sentence_token_features = []
    for token in tokens:
        num_value = None

        if token.pos_ == "NUM":
            num_value = token.text
            token = token.head

        token_features = token2features(token, sentence_id, num_value)
        sentence_token_features.append(token_features)

    return sentence_token_features




if __name__ == "__main__":
    pass