from dataclasses import dataclass
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from . import register_task



@register_task("multilingual_translation_from_pretrained_xlm")
class MultilingualTranslationFromPretrainedXLMTask(TranslationMultiSimpleEpochTask):
    """
    Same as TranslationMultiSimpleEpochTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @classmethod
    def load_dictionary(cls, filename):
        """Load the masked LM dictionary from the filename

        Args:
            filename (str): the filename
        """
        return MaskedLMDictionary.load(filename)
