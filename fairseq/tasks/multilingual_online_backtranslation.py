import os 
import json 
import logging 
from pathlib import Path
from argparse import Namespace
import numpy as np
import math
import random
import math 
from sacrebleu.metrics import BLEU
from typing import Dict, Sequence, Tuple
from collections import OrderedDict, defaultdict

import torch 
import torch.nn.functional as F

from fairseq import metrics, options, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    NoisingDataset,
    PrependTokenDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    data_utils,
    encoders,
    SampledMultiEpochDataset
)
from fairseq.tasks import register_task
from fairseq.metrics import get_active_aggregators
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)

from fairseq.data.multilingual.sampling_method import SamplingMethod, temperature_sampling
from fairseq.tasks.online_backtranslation import PiecewiseLinearFn
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.utils import FileContentsAction
DATASET_TYPES = ["BT", "DENOISE", "MAIN"]
logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4
def _lang_token(lang: str) -> str:
    return f"__{lang}__"


def _lang_token_index(dictionary, lang: str) -> int:
    return dictionary.index(_lang_token(lang))

def def_value():
    return 0

def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return DATASET_TYPES[choiceIndex]
        choiceIndex += 1


def distr(weights_dict, gamma=0.0):
    weights = weights_dict.values()
    theSum = float(sum(weights))
    new_weights = tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)
    weights = {x[0]: x[1] for x in zip(DATASET_TYPES, new_weights)}
    return weights

def value_dict():
    return {'langs' : [], 'prop': []}


def load_sampling_weights(from_file):
    with open(from_file) as f:
        weights = json.load(f)
        
    tgt_samples = defaultdict(def_value)
    src_samples = defaultdict(value_dict)
    for k, v in weights.items():
        src_tgt = k.split(':')[-1]
        src = src_tgt.split('-')[0]
        tgt = src_tgt.split('-')[1]
        tgt_samples[tgt] = tgt_samples[tgt] + v 
        src_samples[tgt]['langs'].append(src)
        src_samples[tgt]['prop'].append(v)
        
    final_src_samples = {}
    for k, v in src_samples.items():
        final_src_samples[k] = {'langs' : [ x for x in v['langs']], 'prop' : [x / sum(v['prop']) for x in v['prop']]}
    
    return tgt_samples, final_src_samples



    

@register_task("online_multilingual_backtranslation")
class MultilingualOnlineBackTranslationTask(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # Generic translation args
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--mono-langs', metavar='MONO_LANGS',
                            help='monolingual languages for training')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # Denoising args
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

        # Backtranslation args
        parser.add_argument('--lambda-bt', default="1.0", type=str, metavar='N',
                            help='back-translation weight')
        parser.add_argument('--lambda-dae', default="1.0", type=str, metavar='N',
                            help='denoising auto-encoder weight')
        parser.add_argument('--lambda-main', default="1.0", type=str, metavar='N',
                            help='main supervised fine tuning weight')
        

        parser.add_argument('--mono-temp', default="1.0", type=str, metavar='N',
                            help='temperature sampling for monolingual data')

        # Evaluation args
        parser.add_argument('--generate-one-by-one', action='store_true',
                            help='generate one sentence at a time for backtranslation')

        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--use-teacher', action='store_true',
                            help='use teacher for backtranslation')
        parser.add_argument('--eval-langs-sep', action = 'store_true')
        parser.add_argument('--score-comet-model', type=str, default=None)
        
        
        
        ## Bandit args
        parser.add_argument('--enable-bandit-sampling', action = 'store_true')
        parser.add_argument('--min-steps-before-sampling', type=int, default = 100)
        parser.add_argument('--bandit-gamma', type = float, default = 0.5)
        parser.add_argument('--bandit-scalar', type = float, default = 1.0)
        parser.add_argument('--mono-temp-file', type = str, default = None )
        
        
  
        
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.mono_langs = args.mono_langs.split(",")
        self.mono_temp_file = args.mono_temp_file 
        


        self.SHOW_SAMPLES_INTERVAL = 1000
        # Start by showing samples
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        self.SHOW_SAMPLES_NUMBER = 5
        if not self.args.enable_bandit_sampling:
            self.lambda_bt = PiecewiseLinearFn.from_string(args.lambda_bt)
            self.lambda_dae = PiecewiseLinearFn.from_string(args.lambda_dae)
            self.lambda_main = PiecewiseLinearFn.from_string(args.lambda_main)
        else: 
            #self.sampling_weights = {"BT": 1.0, "DENOISE": 1.0, "MAIN" : 1.0}
            self.weights = {"BT": 1.0, "DENOISE": 0.0, "MAIN" : 1.0}
            self.prev_reward = 0
            self.gamma = self.args.bandit_gamma
            self.bandit_scalar = self.args.bandit_scalar
            self.reward = 0
            self.total_steps_choice = {"BT": 0, "DENOISE": 0, "MAIN" : 0}
            self.choice = "MAIN"
        self.use_teacher = args.use_teacher



        self.args = args
        self.data = utils.split_paths(self.args.data)
        if len(self.data) == 1:
            shards = list(Path(self.data[0]).glob("shard*"))
            if len(shards) > 0:
                # keep this as strings, since it can also be a manifold path
                old_data = self.data
                self.data = [str(shard) for shard in shards]
                logger.warning(f"Expanded data directory {old_data} to {self.data}")

        self.dictionary = self.dicts["en"]
        


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """


        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )

        return cls(args, langs, dicts, training)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs) -> FairseqDataset:
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_path = self.data[0]
        train_subset = getattr(self.args, "train_subset", None)
        if split == train_subset:
            dataset = self.load_train_dataset(data_path, train_subset)
            self.datasets[train_subset] = dataset
        else:
            # valid/test should always be the same.
            super().load_dataset(getattr(self.args, "valid_subset", None))


    def load_train_dataset(self, data_path: str, train_subset: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        super().load_dataset(train_subset)
        
        data = []
        data.append((f"all-MAIN", self.datasets[train_subset]))
        bt_data = []
        denoise_data = [] 
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            bt_data.append(self.load_bt_dataset(train_path, lang))
            denoise_data.append(self.load_denoise_dataset(train_path, lang))
        sizes = [len(d) for d in bt_data]
        if self.mono_temp_file is None:
            sampling_ratios = temperature_sampling(sizes, float(self.args.mono_temp))
        else:
            tgt_sampling_ratios, src_sampling_ratios = load_sampling_weights(self.mono_temp_file)
            self.src_sampling_ratios = src_sampling_ratios
            sampling_ratios = [tgt_sampling_ratios.get(x) for x in self.mono_langs]
        data.append(("all-BT", SampledMultiEpochDataset(bt_data, sampling_ratios=sampling_ratios )))
        data.append(("all-DENOISE", SampledMultiEpochDataset(denoise_data, sampling_ratios=sampling_ratios)))
        return RoundRobinZipDatasets(OrderedDict(data))
    


    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.dictionary, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        mono_dataset_src = PrependTokenDataset(
            mono_dataset, _lang_token_index(self.dictionary, lang)
        )
        mono_dataset_tgt = PrependTokenDataset(mono_dataset_src, self.dictionary.eos())

        mono_dataset_bt = self._langpair_dataset(mono_dataset_src, mono_dataset_tgt)
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {_lang_token_index(self.dictionary, lang)} "
            f"lang token = {_lang_token(lang)}"
        )

        mono_dataset_bt = self._prepend_lang_bos_to_target(mono_dataset_bt, lang)
        return mono_dataset_bt


    def _langpair_dataset(
        self, src: FairseqDataset, tgt: FairseqDataset
    ) -> LanguagePairDataset:
        return LanguagePairDataset(
            src,
            src.sizes,
            self.dictionary,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target
        )
    
    def _prepend_lang_bos_to_target(
        self, dataset: LanguagePairDataset, lang: str
    ) -> LanguagePairDataset:
        bos = _lang_token_index(self.dicts["en"], lang)
        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.dictionary.eos(),
            new_src_eos=self.dictionary.eos(),
            tgt_bos=self.dictionary.eos(),
            new_tgt_bos=bos,
        )
    
    def load_denoise_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """Classic denoising dataset"""
        dataset = data_utils.load_indexed_dataset(
            data_path, self.dicts["en"], self.args.dataset_impl
        )
        noisy_dataset = NoisingDataset(
            dataset,
            self.dicts["en"],
            seed=1,
            max_word_shuffle_distance=self.args.max_word_shuffle_distance,
            word_dropout_prob=self.args.word_dropout_prob,
            word_blanking_prob=self.args.word_blanking_prob,
        )
        noisy_dataset = PrependTokenDataset(
            noisy_dataset, _lang_token_index(self.dictionary, lang)
        )

        clean_dataset = data_utils.load_indexed_dataset(
            data_path, self.dictionary, self.args.dataset_impl
        )
        denoising_dataset = self._langpair_dataset(noisy_dataset, clean_dataset)
        denoising_dataset = self._prepend_lang_bos_to_target(denoising_dataset, lang)
        return denoising_dataset
    
    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        detok_args =  {}
        self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
        )
        gen_args = json.loads(self.args.eval_bleu_args)
        self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
        )
        return model
    
    def display_samples_once_in_a_while(self, smp):
        self._show_samples_ctr += 1
        if self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            return
        self._show_samples_ctr = 0

        ln = smp["net_input"]["src_tokens"].shape[0]

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.dictionary.string(src_tokens, "sentencepiece")
            tgt_str = self.dictionary.string(tgt_tokens, "sentencepiece")
            logger.info(
                f"\n{i}\t\t[ generated ]  {src_str}\n"
                f"\t\t[ original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def backtranslate_sample(self, smp, other_lang, model) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (from data) __en__ hello world |  __en__ hello world   |
          |--------------------------------------------------------|
        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (generated) __ro__ salut lume  |  __en__ hello world   |
          |--------------------------------------------------------|
        """
        model.eval()
        lang_token = _lang_token_index(self.dictionary, other_lang)
        net_input = smp["net_input"]
        if self.mono_temp_file is not None:
            tgt_langs = [self.dictionary.__getitem__(x).replace('__', '') for x in smp['target'][:, 1].tolist()]
            src_langs = []
            for tgt in tgt_langs:
                langs = self.src_sampling_ratios[tgt]['langs']
                probs =  self.src_sampling_ratios[tgt]['prop']
                src_langs.append(_lang_token_index(self.dictionary, np.random.choice(langs,p =  probs)))

            prefix_tokens = torch.tensor(src_langs,  dtype=net_input["src_tokens"].dtype, device = net_input["src_tokens"].device).reshape((net_input['src_tokens'].shape[0], 1))
        else:
            prefix_tokens = torch.full((net_input['src_tokens'].shape[0], 1), lang_token, dtype=net_input["src_tokens"].dtype, device = net_input["src_tokens"].device )
        src_tokens = net_input['src_tokens'].shape[0]
        generated = self.sequence_generator.generate(
            models=[model], sample=smp, prefix_tokens= prefix_tokens, 
        )
        max_lngth = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_lngth + 1), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0) - 1
            padding_needed = max_lngth - tokens_size
            #tokens = torch.cat([tokens.new([lang_token]), tokens])
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size + 1

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)


    
    def get_other_lang(self, lang):
        # TODO: allow more complex mapping
        if lang != self.mono_langs[0]:
            return self.mono_langs[0]
        if len(self.mono_langs) == 2:
            return self.mono_langs[1]
        #if self.mono_temp_file is None:
        return self.mono_langs[np.random.randint(1, len(self.mono_langs))]
        #else:
        #    src_langs_dict = self.src_sampling_ratios
        #    return np.random.choice(src_langs_dict['langs'], p = src_langs_dict['prop'])
    

    def pick_new_dataset(self, num_updates, valid_loss, maximize = True):
        if num_updates >= self.args.min_steps_before_sampling and self.args.enable_bandit_sampling:
            if sum(self.total_steps_choice.values()) < 1:
                reward = 0
                self.prev_reward = valid_loss
            else:
                reward = (valid_loss - self.prev_reward) * self.bandit_scalar
                if not maximize:
                    reward = -1 * reward
                logger.info(f" previous bleu score {self.prev_reward}, current {valid_loss}, reward {reward}")
                logger.info(f"current reward weight {self.weights}")
                self.prev_reward = valid_loss
            reward = 1.0 * reward / self.weights[self.choice]
            self.weights[self.choice] *= math.exp(reward * self.gamma / 3)
            sampling_weights = distr(self.weights)
            choice = draw(sampling_weights.values())
            while choice != 'DENOISE':
                choice = draw(sampling_weights.values())
            self.choice = choice
            self.total_steps_choice[self.choice] += 1 
            logger.info(f"drawing from {sampling_weights} ... {self.choice}")
            logger.info(f" total draws {self.total_steps_choice}")
        
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, teacher_model=None
    ):

        model.train()
        model.set_num_updates(update_num)
        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)
        
        train_subset = getattr(self.args, "train_subset", None)
        dataset_keys = self.datasets[train_subset].datasets.keys()
        if not self.args.enable_bandit_sampling:
            weights = {
                "BT": self.lambda_bt(update_num),
                "DENOISE": self.lambda_dae(update_num),
                "MAIN": self.lambda_main(update_num) 
            }
        else:
            weights = defaultdict(def_value)
            weights[self.choice] = 1.0
            
        log_keys = {"BT": "bt_", "DENOISE": "dae_", "MAIN" : "main_"}
        for dataset_key in dataset_keys:
            smp = sample[dataset_key]
            _, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            
            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.eval()
                    other_lang = self.get_other_lang("all")
                    self.backtranslate_sample(smp, other_lang, model if not self.use_teacher else teacher_model)
                    self.display_samples_once_in_a_while(smp)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp, teacher_model=teacher_model)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output
    
    def get_bos_token_from_sample(self, sample):
        net_input = sample["net_input"]
        source_lang_token_id = torch.unique(net_input["src_tokens"][:, 0]).item()
        source_lang_token = self.dictionary[source_lang_token_id].replace("_", "")
        target_lang_token_id = _lang_token_index(
            self.dictionary, self.get_other_lang(source_lang_token)
        )

        return target_lang_token_id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.enable_bandit_sampling:
            for choice in ["BT", "MAIN", "DENOISE"]:
                metrics.log_scalar(f"reward_{choice}", self.weights[choice] , round = 3)
                metrics.log_scalar(f"total_steps_{choice}", self.total_steps_choice[choice])

        bt_sample_size = sum(x.get("bt_sample_size", 0) for x in logging_outputs)
        if bt_sample_size:
            bt_loss_sum = sum(x.get("bt_loss", 0) for x in logging_outputs)
            bt_loss_sum *= 1 / bt_sample_size / math.log(2)
            metrics.log_scalar("bt_loss", bt_loss_sum, bt_sample_size, round=3)

            bt_nll_loss_sum = sum(x.get("bt_nll_loss", 0) for x in logging_outputs)
            bt_ntokens = sum(x.get("bt_ntokens", 0) for x in logging_outputs)
            bt_nll_loss_sum *= 1 / bt_ntokens / math.log(2)
            metrics.log_scalar("bt_nll_loss", bt_nll_loss_sum, bt_ntokens, round=3)
            metrics.log_derived(
                "bt_ppl", lambda meters: utils.get_perplexity(meters["bt_nll_loss"].avg)
            )

        dae_sample_size = sum(x.get("dae_sample_size", 0) for x in logging_outputs)
        if dae_sample_size:
            dae_loss_sum = sum(x.get("dae_loss", 0) for x in logging_outputs)
            dae_loss_sum *= 1 / dae_sample_size / math.log(2)
            metrics.log_scalar("dae_loss", dae_loss_sum, dae_sample_size, round=3)

            dae_nll_loss_sum = sum(x.get("dae_nll_loss", 0) for x in logging_outputs)
            dae_ntokens = sum(x.get("dae_ntokens", 0) for x in logging_outputs)
            dae_nll_loss_sum *= 1 / dae_ntokens / math.log(2)
            metrics.log_scalar("dae_nll_loss", dae_nll_loss_sum, dae_ntokens, round=3)
            metrics.log_derived(
                "dae_ppl",
                lambda meters: utils.get_perplexity(meters["dae_nll_loss"].avg),
            )


