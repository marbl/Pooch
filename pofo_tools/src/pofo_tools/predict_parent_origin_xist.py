#!/usr/bin/env python3
"""Predict with saved CpG-only, XIST-covariate, or XIST-interaction models.

Accepts the complete .model.pkl bundle from elasticnet_xist_*.py or a fitted
sklearn Pipeline. Always predicts THROUGH the saved pipeline, never refits.
XIST_diff may be in the dataset or supplied with --xist. Its signed difference
must follow the SAME operand direction as the CpGs and the training data.

XIST TSV keys: Sample; optionally BOTH LeftOpHapLabel and RightOpHapLabel.
Without operand keys, one XIST_diff per Sample is required and assumed to be
oriented to that sample's input row. Mixed operand orientations for a sample
require operand keys. No training-time sample corrections are silently repeated.
"""

import argparse
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

LOG = logging.getLogger(__name__)
META = ['Sample', 'LeftOpHapLabel', 'RightOpHapLabel']
CPG_PATTERN = re.compile(r'(?:chr[0-9A-Za-z]+_[0-9]+_[0-9]+|[0-9]+)')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input-dataset', required=True, type=Path)
    p.add_argument('-M', '--model-pkg', required=True, type=Path,
                   help='Complete fitted Pipeline or new .model.pkl bundle; not classifier-only file')
    p.add_argument('--xist', type=Path, help='TSV with Sample and XIST_diff')
    p.add_argument('-c', '--chr', dest='chrom', default='auto')
    p.add_argument('-o', '--output-predictions', type=Path)
    p.add_argument('-l', '--log-file', type=Path)
    p.add_argument('-m', '--maternal-label')
    p.add_argument('-p', '--paternal-label')
    p.add_argument('-X', '--output-xformed', action='store_true',
                   help='Write actual transformed classifier inputs')
    p.add_argument('--positive-class-label', default='1',
                   help='Model class representing positive origin [1; also matches True]')
    p.add_argument('--positive-class-origin', choices=['right-paternal', 'left-paternal'],
                   default='right-paternal',
                   help='New training scripts use right-paternal; override only for older models')
    p.add_argument('--flip-samples', nargs='*', default=[],
                   help='Explicitly reverse ALL signed input features, including XIST, for these samples')
    a = p.parse_args()
    if bool(a.maternal_label) != bool(a.paternal_label):
        p.error('Provide both -m and -p, or neither.')
    if a.maternal_label and a.maternal_label == a.paternal_label:
        p.error('Maternal and paternal labels must differ.')
    if a.output_predictions is None:
        a.output_predictions = a.input_dataset.with_suffix('.predictions.tsv')
    if a.output_predictions.resolve() == a.input_dataset.resolve():
        p.error('Output must not overwrite the input dataset.')
    return a


def read_table(path):
    # Check header before pandas can silently rename duplicate feature columns.
    import csv
    with open(path, newline='') as handle:
        header = next(csv.reader(handle, delimiter='\t'), [])
    if len(header) != len(set(header)):
        raise ValueError(f'Duplicate column names in {path}')
    return pd.read_csv(path, sep='\t', dtype={'Sample': str,
                       'LeftOpHapLabel': str, 'RightOpHapLabel': str})


def resolve_model(package):
    if isinstance(package, dict):
        if 'pipeline' not in package:
            raise ValueError('Model dictionary must contain a fitted pipeline.')
        pipe = package['pipeline']
        names = package.get('input_features')
        chromosome = package.get('metrics', {}).get('chromosome')
    else:
        pipe, names, chromosome = package, None, None
    if not isinstance(pipe, Pipeline):
        raise ValueError('Supply the COMPLETE fitted Pipeline or .model.pkl bundle, not the classifier alone.')
    if names is None:
        names = getattr(pipe, 'feature_names_in_', None)
    if names is None:
        # The first custom training script did not expose feature_names_in_.
        first = pipe.steps[0][1]
        if hasattr(first, 'selected_cpg_names_'):
            names = list(first.selected_cpg_names_) + ['XIST_diff']
        elif hasattr(first, 'cpg_names_'):
            names = list(first.cpg_names_) + ['XIST_diff']
    if names is None:
        raise ValueError('Saved model lacks input feature names. Re-save with input_features; guessing column order is unsafe.')
    if pipe.steps and hasattr(pipe.steps[0][1], 'selected_cpg_names_') and hasattr(pipe.steps[-1][1], 'coef_'):
        names = restrict_to_nonzero_cpgs(pipe)
    names = [str(c) for c in names]
    if len(names) != len(set(names)):
        raise ValueError('Saved input feature names are not unique.')
    return pipe, names, chromosome


def restrict_to_nonzero_cpgs(pipe):
    """Prune zero-coefficient CpGs from saved CpGScreen elastic-net models."""
    screen = pipe.steps[0][1]
    model = pipe.steps[-1][1]
    cpgs = [str(c) for c in screen.selected_cpg_names_]
    out = [str(c) for c in screen.get_feature_names_out()]
    coef = np.asarray(model.coef_)
    if coef.ndim != 2 or coef.shape[0] != 1 or coef.shape[1] != len(out):
        raise ValueError(
            f'Model has {coef.size} coefficients but CpGScreen outputs {len(out)} features.'
        )

    coef = coef[0]
    active = set()
    for name, value in zip(out, coef):
        if abs(value) <= 1e-7:
            continue
        cpg = name
        for suffix in ('__times_abs_XIST', '__abs_times_XIST'):
            if cpg.endswith(suffix):
                cpg = cpg[:-len(suffix)]
                break
        if cpg != 'XIST_diff':
            active.add(cpg)

    keep_cpgs = [c for c in cpgs if c in active]
    keep_out = [
        name for name in out
        if name == 'XIST_diff' or name.removesuffix('__times_abs_XIST').removesuffix('__abs_times_XIST') in active
    ]
    screen.feature_names_out_ = np.array(keep_out, dtype=object)
    if hasattr(screen, 'selected_mask_'):
        screen.selected_mask_ = np.isin([str(c) for c in screen.cpg_names_], keep_cpgs)
    out_index = {name: i for i, name in enumerate(out)}
    keep_idx = [out_index[name] for name in keep_out]
    screen.selected_cpg_names_ = np.array(keep_cpgs, dtype=object)
    model.coef_ = coef[keep_idx][None, :]
    model.n_features_in_ = len(keep_idx)
    return keep_cpgs + ['XIST_diff']


def attach_xist(data, path):
    cov = read_table(path)
    if not {'Sample', 'XIST_diff'} <= set(cov.columns):
        raise ValueError('--xist requires Sample and XIST_diff columns.')
    operand_keys = ['LeftOpHapLabel', 'RightOpHapLabel']
    present = [k in cov.columns for k in operand_keys]
    if any(present) and not all(present):
        raise ValueError('XIST file must provide both operand labels or neither.')
    keys = ['Sample'] + (operand_keys if all(present) else [])
    if not all(present):
        orientations = data[['Sample'] + operand_keys].drop_duplicates()
        if orientations.Sample.duplicated().any():
            raise ValueError('Multiple operand orientations for one sample: include both operand labels in --xist.')
    if cov.duplicated(keys).any():
        raise ValueError(f'Duplicate XIST keys: {keys}')
    merged = data.merge(cov[keys + ['XIST_diff']], how='left', on=keys,
                        validate='many_to_one', sort=False, suffixes=('', '__external'))
    if 'XIST_diff__external' in merged:
        embedded = pd.to_numeric(merged.XIST_diff, errors='raise').to_numpy(dtype=float)
        external = pd.to_numeric(merged.XIST_diff__external, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(external).all() or not np.allclose(embedded, external, rtol=1e-7, atol=1e-10):
            raise ValueError('Embedded XIST_diff and --xist values differ or lack matching rows.')
        merged = merged.drop(columns='XIST_diff__external')
    return merged


def positive_index(classes, label):
    matches = []
    for i, value in enumerate(classes):
        match = str(value) == label
        if isinstance(value, (bool, np.bool_)):
            match = match or label == str(int(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            try:
                match = match or float(label) == float(value)
            except ValueError:
                pass
        if match:
            matches.append(i)
    if len(classes) != 2 or len(matches) != 1:
        raise ValueError(f'Cannot identify positive class {label!r} in model classes {classes!r}')
    return matches[0]


def main():
    args = parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=handlers)
    # New models embed custom classes with cloudpickle, readable by pickle.load.
    with args.model_pkg.open('rb') as handle:
        package = pickle.load(handle)
    pipe, features, trained_chrom = resolve_model(package)
    data = read_table(args.input_dataset)
    if data.empty or not set(META) <= set(data.columns):
        raise ValueError(f'Nonempty input must contain {META}')
    if data[META].isna().any().any():
        raise ValueError('Sample and operand labels must not be missing.')
    if (data.LeftOpHapLabel == data.RightOpHapLabel).any():
        raise ValueError('Left and right operand labels must differ in every row.')
    chrom = args.chrom
    if chrom.lower() == 'auto':
        matches = re.findall(r'chr[0-9A-Za-z]+', args.input_dataset.name)
        chrom = matches[-1] if matches else trained_chrom
    if trained_chrom and chrom and chrom != trained_chrom:
        raise ValueError(f'Model chromosome {trained_chrom} differs from input chromosome {chrom}.')
    if args.xist:
        data = attach_xist(data, args.xist)
    missing = [c for c in features if c not in data.columns]
    if missing:
        hint = ' Supply --xist or an embedded XIST_diff column.' if 'XIST_diff' in missing else ''
        raise ValueError(
            f'Missing {len(missing)} model inputs from {args.input_dataset.resolve()}: '
            f'{missing[:15]}.{hint}'
        )
    # Exact stored order; ignore extra CpGs and never manually recreate interactions.
    X = data.loc[:, features].apply(pd.to_numeric, errors='raise').astype(float)
    finite = np.isfinite(X.to_numpy()).all(axis=1)
    if not finite.all():
        raise ValueError(f'Missing/nonfinite model inputs for samples: {data.loc[~finite, "Sample"].tolist()[:15]}')
    if args.flip_samples:
        flip = data.Sample.isin(args.flip_samples)
        X.loc[flip, :] = -X.loc[flip, :]
        LOG.info('Explicit sign correction applied to %d rows.', int(flip.sum()))
    LOG.info('Predicting %d rows with %d raw features through saved pipeline.', len(X), X.shape[1])
    probabilities = np.asarray(pipe.predict_proba(X))
    idx = positive_index(np.asarray(pipe.classes_), args.positive_class_label)
    if probabilities.shape != (len(X), 2) or not np.isfinite(probabilities).all():
        raise ValueError('Model returned invalid binary probabilities.')
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Model probabilities must be in [0,1] and sum to one.')
    p_right = probabilities[:, idx]
    if args.positive_class_origin == 'left-paternal':
        p_right = 1 - p_right
    right_paternal = p_right >= 0.5
    # Keep metadata in input order; remove raw CpG feature columns.
    metadata = [c for c in data.columns if c not in features and not CPG_PATTERN.fullmatch(c)]
    metadata = list(dict.fromkeys(META + metadata))
    out = data.loc[:, metadata].copy()
    if 'XIST_diff' in features:
        out['XIST_diff'] = X['XIST_diff'].to_numpy()
    out['LeftHapPredictedOrigin'] = np.where(right_paternal, 'Maternal', 'Paternal')
    out['RightHapPredictedOrigin'] = np.where(right_paternal, 'Paternal', 'Maternal')
    out['PredictProb_Maternal-Paternal'] = p_right
    out['PredictProb_Paternal-Maternal'] = 1 - p_right
    out['PredictionCorrect'] = 'Unknown'
    if args.maternal_label:
        valid = ((data.LeftOpHapLabel == args.maternal_label) & (data.RightOpHapLabel == args.paternal_label)) | (
            (data.LeftOpHapLabel == args.paternal_label) & (data.RightOpHapLabel == args.maternal_label))
        if not valid.all():
            raise ValueError('Every row must contain exactly one maternal and one paternal label matching -m/-p.')
        truth = (data.RightOpHapLabel == args.paternal_label).to_numpy()
        out['PredictionCorrect'] = truth == right_paternal
        LOG.info('Accuracy: %.4f; Brier score: %.4f', accuracy_score(truth, right_paternal), brier_score_loss(truth, p_right))
        LOG.info('Confusion matrix (rows truth, columns prediction; Right=Paternal classes 0,1):\n%s',
                 confusion_matrix(truth, right_paternal, labels=[0, 1]))
        if len(np.unique(truth)) == 2:
            LOG.info('AUC: %.4f', roc_auc_score(truth, p_right))
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    if args.output_xformed:
        if len(pipe.steps) > 1:
            preprocessor = pipe[:-1]
            transformed = preprocessor.transform(X)
            try:
                names = preprocessor.get_feature_names_out()
            except (AttributeError, TypeError):
                names = [f'transformed_{i}' for i in range(transformed.shape[1])]
        else:
            transformed, names = X.to_numpy(), features
        if sparse.issparse(transformed):
            transformed = transformed.toarray()
        intermediate = pd.concat([data[META].reset_index(drop=True),
                                  pd.DataFrame(np.asarray(transformed), columns=names)], axis=1)
        path = args.output_predictions.with_suffix('.intermXformed.tsv')
        intermediate.to_csv(path, sep='\t', index=False)
        LOG.info('Wrote transformed model inputs: %s', path)
    out.to_csv(args.output_predictions, sep='\t', index=False, float_format='%.10g')
    LOG.info('Wrote predictions: %s', args.output_predictions)


if __name__ == '__main__':
    main()
