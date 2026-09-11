#!/usr/bin/env python3
"""Check that a correctly predicted peptide matches its own ground-truth label.

Two properties of ground_truth_mapper have to hold together for this, and either alone is not enough:

1. N_TERM_MOD_PATTERN accepts an N-terminal modification written with or without a leading `n`, since labels
   carry whichever notation their search engine emits. MSFragger uses the `n` form: every N-terminally
   modified sequence in data/human_mAb_trypsin/labels.csv (74 of 2215) and data/human_mAb_aspn/labels.csv
   (5 of 432) is written that way, and the remaining sequences have no N-terminal modification. An
   unaccepted `n` survives as a literal token, which has no entry in AA_MASSES and is scored with mass 0.
   Those are the two label files available locally; datasets searched with MS-GF+ or Comet are not checked
   here.

2. format_sequence folds the modification onto the first residue, matching what the prediction side does
   (utils.ptms_to_delta_mass rewrites "[UNIMOD:385]-SGGSAPYGK" to "S[-17.026549]GGSAPYGK"). A "[+mod]-"
   prefix on one side against a modified first residue on the other is 26 tokens against 25, and
   aa_match_batch scores a perfectly predicted peptide as 24/26 amino acids with EXACT = False.

The assertions target the end property - a correct prediction matches its label - rather than the shape of
either string, so the two separate implementations of the folding cannot drift apart unnoticed.

Runs standalone (no pytest required):
    python test_ground_truth_notation.py
"""

import sys

# (label as distributed, the same peptide predicted correctly in the wrappers' UNIMOD notation)
CASES = [
    ("n[57.0215]SNWEAGNTFTC[57.0215]SVLHEGLHNHHTEK",
     "[UNIMOD:4]-SNWEAGNTFTC[UNIMOD:4]SVLHEGLHNHHTEK"),
    ("[57.0215]SNWEAGK", "[UNIMOD:4]-SNWEAGK"),                       # same, without the `n`
    ("MSGEC[57.0215]APNVSVSVSTSHTTISGGGSR",                            # residue-attached only
     "MSGEC[UNIMOD:4]APNVSVSVSTSHTTISGGGSR"),
    ("SSSSGSVGESSSK", "SSSSGSVGESSSK"),                                # unmodified
]


def test_notation_accepts_both_label_forms(mapper):
    """The two label notations describe the same peptide and must format identically."""
    assert mapper.format_sequence("n[57.0215]SNWEAGK") == mapper.format_sequence("[57.0215]SNWEAGK")


def test_n_term_mod_is_folded_not_left_as_prefix(mapper):
    """A `[mod]-` prefix would mismatch the prediction side, which folds onto residue 1."""
    out = mapper.format_sequence("n[57.0215]SNWEAGK")
    assert not out.startswith("["), f"still a prefix form: {out!r}"
    assert out.startswith("S[+57.0215]"), out


def test_mod_on_modified_first_residue_sums(mapper):
    """N-terminal modification plus a modified first residue must add, as utils does."""
    out = mapper.format_sequence("n[42.0106]M[15.9949]GGGSAPYGK")
    assert out.startswith("M[+58.0055]"), out


def test_unmodified_and_internal_mods_untouched(mapper):
    assert mapper.format_sequence("SSSSGSVGESSSK") == "SSSSGSVGESSSK"
    assert mapper.format_sequence("MSGEC[57.0215]APNK") == "MSGEC[+57.0215]APNK"


def test_correct_prediction_matches_its_label(mapper, utils, aa_match_batch, aa_masses):
    """The property that matters: a perfect prediction scores as an exact match."""
    for label, prediction in CASES:
        truth = mapper.format_sequence(label)
        predicted, _ = utils.ptms_to_delta_mass(prediction, ",".join(["-0.1"] * 60))
        matches, n_pred, n_true = aa_match_batch([predicted], [truth], aa_masses)
        residues, exact = matches[0][0], matches[0][1]
        assert n_pred == n_true, (
            f"tokenisation disagrees for {label!r}: prediction {predicted!r} has {n_pred} tokens, "
            f"ground truth {truth!r} has {n_true}"
        )
        assert int(residues.sum()) == len(residues), (
            f"only {int(residues.sum())}/{len(residues)} residues matched for {label!r}"
        )
        assert bool(exact), f"no exact match for {predicted!r} against {truth!r}"


def main():
    sys.path.insert(0, ".")
    from evaluation import ground_truth_mapper as mapper

    # These four need no heavy dependencies.
    light = [
        test_notation_accepts_both_label_forms,
        test_n_term_mod_is_folded_not_left_as_prefix,
        test_mod_on_modified_first_residue_sums,
        test_unmodified_and_internal_mods_untouched,
    ]
    failures = []
    for check in light:
        try:
            check(mapper)
            print(f"  pass  {check.__name__}")
        except AssertionError as exc:
            failures.append((check.__name__, exc))
            print(f"  FAIL  {check.__name__}: {exc}")

    # The end-to-end match needs numpy and pyteomics via evaluation.utils. Report rather than fail if absent,
    # so the notation checks above still run in a bare environment.
    try:
        from evaluation import utils
        from evaluation.metrics import aa_match_batch
        from evaluation.token_masses import AA_MASSES
    except Exception as exc:  # noqa: BLE001
        print(f"  skip  test_correct_prediction_matches_its_label "
              f"(evaluation.utils unavailable: {type(exc).__name__}: {exc})")
    else:
        try:
            test_correct_prediction_matches_its_label(mapper, utils, aa_match_batch, AA_MASSES)
            print("  pass  test_correct_prediction_matches_its_label")
        except AssertionError as exc:
            failures.append(("test_correct_prediction_matches_its_label", exc))
            print(f"  FAIL  test_correct_prediction_matches_its_label: {exc}")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
