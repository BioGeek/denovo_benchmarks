"""Methods to convert ground truth labels to the common data format."""

import re


REPLACEMENTS = []
PTM_PATTERN = r"([A-Z])\[([0-9.+-]+)\]" # find AAs with PTMs

# The `n` prefix is optional because labels are read exactly as distributed, and search engines differ in
# how they write an N-terminal modification. MSFragger uses n[57.0215]EAKVQWK..., the form taken by every
# N-terminally modified sequence in human_mAb_trypsin (74 of 2215) and human_mAb_aspn (5 of 432); the
# remaining sequences in both files carry no N-terminal modification. An unmatched `n` survives as a literal
# token, which has no entry in AA_MASSES and is scored with mass 0.
#
# Normalising the notation where labels are generated would be the better place, but cannot replace this:
# labels already distributed are read as they are, and regenerating any of them needs per-dataset configs
# that the benchmark repository does not ship, so such a fix is unverifiable from outside. Folding the
# modification onto the first residue, below, is needed either way - that mismatch is between this module
# and evaluation.utils, not in the source notation.
N_TERM_MOD_PATTERN = r"^n?\[([0-9.+-]+)\]" # find N-term modifications, with or without the `n` prefix

# format_sequence folds the N-terminal modification onto the first residue rather than leaving it as a
# `[mod]-` prefix, so that labels tokenise exactly as predictions do: evaluate.py passes predictions through
# utils.ptms_to_delta_mass, whose N_TERM_PATTERN rewrites "[UNIMOD:385]-SGGSAPYGK" to "S[-17.026549]GGSAPYGK".
# The two sides must agree, because aa_match_batch walks them token by token - a prefix on one side and a
# modified first residue on the other is a 26-token sequence against a 25-token one, and a perfectly
# predicted peptide then scores 24/26 amino acids with no exact match instead of 25/25.
FIRST_AA_AFTER_N_TERM_MOD = r"^\[([0-9.+-]+)\]-([A-Z])(?:\[([0-9.+-]+)\])?"

def _transform_match_ptm(match: re.Match) -> str:
    """
    Transform representation of amino acids substring matching
    the PTM pattern.
    Expects PTMs in ProForma notation, e.g. 'M[UNIMOD:35]'.

    Parameters
    ----------
    match : re.Match
        Substring matching the PTM pattern.

    Returns
    -------
    transformed_match : str
        Transformed PTM pattern representation.
    """
    aa, ptm = match.group(1), match.group(2)

    if not ptm.startswith("-"):
        ptm = "+" + ptm
    return f"{aa}[{ptm}]"

def _transform_match_n_term_mod(match: re.Match) -> str:
    """
    Transform representation of peptide substring matching
    the N-term modification pattern.
    `n[+n_mod]PEP` -> `[+n_mod]-PEP`
    
    TODO.
    """
    ptm = match.group(1)
    
    if not ptm.startswith("-"):
        ptm = "+" + ptm
    return f"[{ptm}]-"


def _fold_n_term_mod_onto_first_aa(match: re.Match) -> str:
    """Move a `[+mod]-` prefix onto the first residue: `[+m]-PEP` -> `P[+m]EP`.

    Mirrors utils._transform_match_n_term, which does this on the prediction side. Implemented separately
    rather than imported because this module is deliberately stdlib-only, while evaluation.utils builds a
    Unimod database at import time. test_ground_truth_notation.py asserts that a label and the corresponding
    prediction MATCH under aa_match_batch, which is the property that matters and keeps the two
    implementations from drifting apart.

    format_sequence is not idempotent - applied twice, the already-signed "[+57.02]" is matched again and
    becomes "[++57.02]--" - so it must be applied exactly once per label.
    """
    n_term_mod, first_aa, first_aa_ptm = match.group(1), match.group(2), match.group(3)
    if first_aa_ptm is not None:
        # The first residue is itself modified: the two deltas add.
        combined = float(first_aa_ptm) + float(n_term_mod)
        return f"{first_aa}[{combined:+}]"
    return f"{first_aa}[{n_term_mod}]"


def format_sequence(sequence: str) -> str:
    """
    Convert peptide sequence to the common output data format.

    Parameters
    ----------
    sequence : str
        Peptide sequence in the original ground truth format.

    Returns
    -------
    transformed_sequence : str
        Peptide sequence in the common output data format.
    """

    # direct (token-to-token) replacements (if any)
    for repl_args in REPLACEMENTS:
        sequence = sequence.replace(*repl_args)

    # transformation of PTM notation:
    # represent in ProForma delta mass notation PE[+ptm]P
    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)
    
    # transform n-term modification notation
    # represent in ProForma delta mass notation [+n_term_mod]-PEP
    sequence = re.sub(N_TERM_MOD_PATTERN, _transform_match_n_term_mod, sequence)

    # then fold that prefix onto the first residue, so the ground truth is tokenised the same way the
    # prediction side is by utils.ptms_to_delta_mass. See FIRST_AA_AFTER_N_TERM_MOD above.
    sequence = re.sub(FIRST_AA_AFTER_N_TERM_MOD, _fold_n_term_mod_onto_first_aa, sequence)

    return sequence
