import textattack
import warnings
from nltk.tokenize import sent_tokenize

available_textattack_perturbations = {
    "wordnet": textattack.augmentation.recipes.WordNetAugmenter,
    "charswap": textattack.augmentation.recipes.CharSwapAugmenter,
    "embedding": textattack.augmentation.recipes.EmbeddingAugmenter,
    "back_trans": textattack.augmentation.recipes.BackTranslationAugmenter,
    "back_transcription": textattack.augmentation.recipes.BackTranscriptionAugmenter
}
word_level_transformations = {"wordnet", "charswap", "embedding"}
sentence_level_transformations = {"back_trans", "back_transcription"}

def produce_perturbations(task_nl, perturbation_recipe, pct_words_to_swap, perturbations_number) -> list[str]:
    augmenter = available_textattack_perturbations[perturbation_recipe](
                                                pct_words_to_swap=pct_words_to_swap, 
                                                transformations_per_example=perturbations_number)

    if perturbation_recipe in word_level_transformations:
        res = augmenter.augment(task_nl)
    elif perturbation_recipe in sentence_level_transformations:
        warnings.warn("There are known issues with the number of transformations produced by this recipes. See https://github.com/QData/TextAttack/issues/800")
        sentences = sent_tokenize(task_nl)
        perturbed_sentences = [augmenter.augment(s) for s in sentences]
        res = [" ".join(ls) for ls in zip(*perturbed_sentences)]
    else:
        raise ValueError("Transformation not recognized.")
    return res