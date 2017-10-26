from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree


def get_tagged_tree(text: str):
    return ne_chunk(pos_tag(word_tokenize(text)))


def get_named_entities_from_text(text: str):
    return get_named_entities_from_tree(get_tagged_tree(text))


def get_named_entities_from_tree(tree: Tree):
    output = []
    for chunk in tree:
        if type(chunk) == Tree:
            output.extend(get_named_entities_from_tree(chunk))
        else:
            token, label = chunk
            if 'NN' in label:
                output.append(token)
    return list(set(output))
