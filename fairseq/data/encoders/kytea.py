from dataclasses import dataclass, field
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass
from typing import List
import six
import re 

@dataclass
class KyteaTokenizerConfig(FairseqDataclass):
    model_file: str = field(default="en", metadata={"help": "where is the model file located"})
    decode_only: bool = field(default = False,  metadata={"help": "detokenize only"})
    encode_only: bool = field(default = False,  metadata={"help": "tokenize only"})


def to_unicode(unicode_or_str):
    if six.PY3:
        return unicode_or_str
    if isinstance(unicode_or_str, str):
        value = unicode_or_str.decode('utf-8')
    else:
        value = unicode_or_str
    return value  # Instance of unicode


def force_to_unicode(s):
    """ Returns the joined string if s is a list. """
    if isinstance(s, list):
        s = " ".join(s)
    assert isinstance(s, six.string_types)
    return to_unicode(s)


def chinese_deseg(words):
    """ Recovers the result of `tokenize(words)`.

    Args:
        words: A list of strings, i.e. tokenized text.

    Returns: The recovered sentence string.
    """
    words = force_to_unicode(words)
    re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
    re_final_comma = re.compile("\.$")
    
    words = re_space.sub("", words)
    # words = words.replace(",", u"\uFF0C")
    words = re_final_comma.sub(u"\u3002", words)
    return words


@register_tokenizer("kytea", dataclass=KyteaTokenizerConfig)
class KyteaTokenizer(object):
    def __init__(self, cfg: KyteaTokenizerConfig):
        try: 
            from Mykytea import Mykytea
        except: 
            raise ImportError 
        self.cfg = cfg
        _param = u"-model {}".format(cfg.model_file)
        self._kt = Mykytea(_param)

    def encode(self, text: str) -> str:
        if self.cfg.decode_only:
            return text
        w_list = []
        for w in self._kt.getWS(text):
            w_list.append(w)
        return u" ".join(w_list)

 
    def decode(self, text:str) -> str:
        if self.cfg.encode_only:
            return text
        res = chinese_deseg(text.split(" "))
        res = re.sub(r" +", u" ", res)
        res = ''.join(res.split())
        return res
