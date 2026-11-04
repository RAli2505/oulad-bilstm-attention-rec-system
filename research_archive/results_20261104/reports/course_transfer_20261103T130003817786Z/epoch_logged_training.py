def fit_predict(name, fit, val, held, test, directory: Path, seed: int, config: dict):
    """One fold, one model. Persist complete parameters, warnings and predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(config['device'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    trace = []
    parameters = None
    risk_held = risk_test = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name in CLASSICAL:
            model = classical_model(name, seed, config['threads'], config.get('mlp_iterations', 200))
            model.set_params(**config.get('model_params', {}))
            fit_start = time.perf_counter()
            model.fit(fit['flat'], fit['y'])
            fit_seconds = time.perf_counter() - fit_start
            score_fit = _score(model, fit['flat'])
            score_val = _score(model, val['flat'])
            score_held = _score(model, held['flat'])
            score_test = _score(model, test['flat']) if test is not None else None
            hyperparameters = model.get_params()
            if hasattr(model, 'coefs_'):
                parameters = sum((a.size for a in model.coefs_)) + sum((a.size for a in model.intercepts_))
            if hasattr(model, 'loss_curve_'):
                trace = [{'iteration': i + 1, 'loss': float(v)} for i, v in enumerate(model.loss_curve_)]
            joblib.dump(model, directory / 'model.joblib')
        else:
            model = make_model(name, fit['x'].shape[2], fit['s'].shape[1], fit['x'].shape[1]).to(device)
            apply_dropout(model, config.get('dropout_scale', 1.0))
            parameters = sum((p.numel() for p in model.parameters()))
            train_data, val_data = (_tensors(fit, device), _tensors(val, device))
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0), fused=device.type == 'cuda')
            bce = nn.BCEWithLogitsLoss()
            best_loss, best_state, bad, best_epoch = (float('inf'), None, 0, 0)
            probe_ids_np = np.random.default_rng(seed).choice(len(fit['y']), min(512, len(fit['y'])), replace=False)
            probe_ids = torch.as_tensor(probe_ids_np, device=device)
            probe = {k: v[probe_ids] for k, v in train_data.items()}
            for epoch in range(config['epochs']):
                epoch_start = time.perf_counter()
                model.train()
                order = torch.randperm(len(fit['y']), device=device)
                total = torch.zeros(3, device=device)
                batches = batch_indices(order, config['batch_size'])
                for ids in batches:
                    optimizer.zero_grad(set_to_none=True)
                    logit, risk = model(train_data['x'][ids], train_data['s'][ids], train_data['weekly'][ids])
                    loss_bce = bce(logit, train_data['y'][ids])
                    loss_cox = breslow_loss(risk, train_data['time'][ids], train_data['event'][ids]) if name in COX else logit.sum() * 0
                    loss = loss_bce + config['cox_weight'] * loss_cox
                    loss.backward()
                    optimizer.step()
                    total += torch.stack([loss.detach(), loss_bce.detach(), loss_cox.detach()])
                score_val, _ = _predict(model, val_data)
                from sklearn.metrics import log_loss, roc_auc_score
                monitor = float(log_loss(val['y'], np.clip(expit(score_val), 1e-07, 1 - 1e-07)))
                losses = (total / len(batches)).cpu().tolist()
                if not np.isfinite([monitor, *losses]).all():
                    raise FloatingPointError(f'Nonfinite epoch loss in {name}')
                probe_score, _ = _predict(model, probe)
                probe_probability = np.clip(expit(probe_score), 1e-07, 1 - 1e-07)
                trace.append({'epoch': epoch + 1, 'train_loss': losses[0], 'train_bce': losses[1], 'train_cox_minibatch': losses[2], 'val_log_loss_raw': monitor, 'train_probe_log_loss_raw': float(log_loss(fit['y'][probe_ids_np], probe_probability, labels=[0, 1])), 'train_probe_auc_raw': float(roc_auc_score(fit['y'][probe_ids_np], probe_score)) if len(np.unique(fit['y'][probe_ids_np])) == 2 else None, 'train_probe_n': len(probe_ids_np), 'val_auc_raw': float(roc_auc_score(val['y'], score_val)), 'seconds': time.perf_counter() - epoch_start})
                print(f"{name} seed {seed} epoch {epoch + 1}/{config['epochs']}: val_loss={monitor:.6f}; epoch_seconds={trace[-1]['seconds']:.1f}; remaining_epoch_budget={config['epochs'] - epoch - 1}", flush=True)
                if monitor < best_loss - config['min_delta']:
                    best_loss, best_state, bad, best_epoch = (monitor, copy.deepcopy(model.state_dict()), 0, epoch + 1)
                else:
                    bad += 1
                if bad >= config['patience']:
                    break
            model.load_state_dict(best_state)
            fit_seconds = sum((row['seconds'] for row in trace))
            score_val, _ = _predict(model, val_data)
            del train_data, val_data, optimizer, probe
            held_data = _tensors(held, device)
            score_held, raw_risk_held = _predict(model, held_data)
            del held_data
            score_test = raw_risk_test = None
            if test is not None:
                test_data = _tensors(test, device)
                score_test, raw_risk_test = _predict(model, test_data)
                del test_data
            risk_stats = None
            fit_data = _tensors(fit, device)
            score_fit, reference_risk = _predict(model, fit_data)
            del fit_data
            if name in COX:
                center, scale = (float(reference_risk.mean()), max(float(reference_risk.std()), 1e-06))
                risk_stats = {'training_mean': center, 'training_std': scale}
                risk_held = (raw_risk_held - center) / scale
                risk_test = (raw_risk_test - center) / scale if test is not None else None
            hyperparameters = {**config, 'model_name': name, 'features': fit['x'].shape[2], 'effective_cox_weight': config['cox_weight'] if name in COX else 0.0, 'static_features': fit['s'].shape[1], 'days': fit['x'].shape[1], 'best_epoch': best_epoch, 'best_val_log_loss_raw': best_loss, 'network': str(model), 'parameter_count': parameters, 'risk_standardization': risk_stats}
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'parameters': hyperparameters}, directory / 'model.pt')
        calibration = Calibration().fit(score_val, val['y'])
        joblib.dump(calibration, directory / 'calibration.joblib')
        pred = {'held_score': score_held, 'held_probability': calibration.predict(score_held), 'val_score': score_val, 'val_target': val['y'], 'fit_score': score_fit, 'fit_target': fit['y']}
        if test is not None:
            pred.update(test_score=score_test, test_probability=calibration.predict(score_test))
        if risk_held is not None:
            pred['held_risk'] = risk_held
            if test is not None:
                pred['test_risk'] = risk_test
        np.savez_compressed(directory / 'predictions.npz', **pred)
        from .metrics import metrics, calibration_stats
        raw_fit, raw_val = (metrics(fit['y'], expit(score_fit)), metrics(val['y'], expit(score_val)))
        prevalence = float(np.mean(fit['y']))
        from sklearn.metrics import log_loss
        null_loss = float(log_loss(fit['y'], np.full(len(fit['y']), prevalence)))
        diagnostics = {'train_raw': raw_fit, 'validation_raw': raw_val, 'val_minus_train_log_loss': raw_val['log_loss'] - raw_fit['log_loss'], 'train_minus_val_auc': raw_fit['roc_auc'] - raw_val['roc_auc'], 'null_train_log_loss': null_loss, 'training_log_loss_improvement_over_null': null_loss - raw_fit['log_loss'], 'training_beats_null_log_loss': raw_fit['log_loss'] < null_loss, 'epochs_ran': len(trace) if name not in CLASSICAL else None, 'epoch_budget_exhausted': len(trace) >= config['epochs'] if name not in CLASSICAL else None, 'best_epoch': hyperparameters.get('best_epoch') if name not in CLASSICAL else None, 'best_epoch_at_budget': best_epoch == config['epochs'] if name not in CLASSICAL else None, 'last_val_minus_best_val_log_loss': trace[-1]['val_log_loss_raw'] - best_loss if name not in CLASSICAL else None, 'mlp_iteration_budget_exhausted': bool(getattr(model, 'n_iter_', 0) >= config.get('mlp_iterations', 200)) if name == 'mlp' else None, 'optimizer_iterations': np.asarray(model.n_iter_).tolist() if name in CLASSICAL and hasattr(model, 'n_iter_') else None, 'interpretation': 'Measured gaps and budget flags are evidence, not automatic diagnoses of over/underfitting.'}
        calibration_rows, bin_rows = ([], [])
        for split_name, label, score in [('train', fit['y'], score_fit), ('validation', val['y'], score_val), ('held', held['y'], score_held)]:
            for stage, prob in [('raw', expit(score)), ('calibrated', calibration.predict(score))]:
                measures, bins = calibration_stats(label, prob)
                calibration_rows.append({'split': split_name, 'stage': stage, **measures})
                bin_rows.extend(({'split': split_name, 'stage': stage, **b} for b in bins))
        import pandas as pd
        pd.DataFrame(calibration_rows).to_csv(directory / 'calibration_metrics.csv', index=False)
        pd.DataFrame(bin_rows).to_csv(directory / 'calibration_bins.csv', index=False)
        if trace:
            pd.DataFrame(trace).to_csv(directory / 'learning_curve.csv', index=False)
            if name not in CLASSICAL and config.get('plot_learning_curves', True):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                curve = pd.DataFrame(trace)
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].plot(curve.epoch, curve.train_probe_log_loss_raw, label='Train probe (eval mode)')
                axes[0].plot(curve.epoch, curve.val_log_loss_raw, label='Validation')
                axes[1].plot(curve.epoch, curve.train_probe_auc_raw, label='Train probe (eval mode)')
                axes[1].plot(curve.epoch, curve.val_auc_raw, label='Validation')
                for ax, ylabel in zip(axes, ['Raw BCE / log loss', 'Raw ROC-AUC']):
                    ax.axvline(best_epoch, linestyle=':', color='grey', label='Selected epoch')
                    ax.set(xlabel='Epoch', ylabel=ylabel)
                    ax.legend(fontsize=8)
                fig.suptitle(name)
                fig.tight_layout()
                fig.savefig(directory / 'learning_curve.png', dpi=160)
                plt.close(fig)
        (directory / 'fit_diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        report = {'model': name, 'seed': seed, 'parameters': hyperparameters, 'n_fit': len(fit['y']), 'n_validation': len(val['y']), 'n_held': len(held['y']), 'seconds': time.perf_counter() - start, 'parameter_count': parameters, 'fit_with_monitoring_seconds': fit_seconds, 'input_static_features': fit['s'].shape[1], 'input_activity_types': fit['x'].shape[2], 'input_daily_steps': fit['x'].shape[1], 'input_classical_features': fit['flat'].shape[1], 'fit_class_counts': np.bincount(fit['y'], minlength=2).tolist(), 'validation_class_counts': np.bincount(val['y'], minlength=2).tolist(), 'fit_withdrawal_events': int(fit['event'].sum()), 'validation_withdrawal_events': int(val['event'].sum()), 'calibration': {'method': 'StandardScaler(score) + Platt LogisticRegression C=1', 'fit_partition': 'inner_validation', 'coefficient': float(calibration.model.coef_[0, 0]), 'intercept': float(calibration.model.intercept_[0]), 'score_mean': float(calibration.scaler.mean_[0]), 'score_std': float(calibration.scaler.scale_[0])}, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None, 'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None, 'trace': trace, 'diagnostics': diagnostics, 'warnings': [str(w.message) for w in caught]}
    (directory / 'training.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    if device.type == 'cuda':
        del model
        torch.cuda.empty_cache()
    return (pred, report)