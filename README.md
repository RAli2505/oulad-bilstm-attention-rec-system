# OULAD: модели прогнозирования и сравнительные эксперименты

Самостоятельный исследовательский репозиторий без зависимости от агентов и оркестратора.

## Состав

- `models/pred_oulad.py`: LSTM, BiLSTM + Attention, TCN + BiLSTM + Attention и ансамбли с LightGBM/XGBoost. Динамические признаки объединяются со статическими; обучение использует BCE + Cox loss.
- `results/oulad/pred_metrics*.json`: исходные сохранённые результаты, включая окна наблюдения, абляции и несколько seed.
- `experiments/hybrid-attention/`: отдельный набор сравнительных экспериментов — MLP, Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, нейронная гибридная модель, кросс-валидация и абляции. Сохранены notebook, таблицы и графики.
- `models/oulad_hybrid_attention.py`: адаптер вывода гибридной модели.
- `data/oulad_loader.py`: подготовка OULAD; локальные raw/processed/splits скопированы и исключены из Git.
- `tests/`: перенесённые тесты модели и метрик.
- `docs/migration_manifest.json`: происхождение файлов и SHA-256 до/после переноса.
- `docs/CROSS_VALIDATION_RU.md`: сохранённый протокол 5-fold CV и таблица mean ± std для всех сравнительных моделей.

## Сохранённые результаты

| Вариант | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| BiLSTM + Attention | 0.87497486 | 0.83497868 | 0.12910403 |
| LSTM + BiLSTM Attention + LightGBM | 0.87860336 | 0.82281372 | 0.12720563 |

Это исторические метрики, а не результат повторного обучения в новом репозитории. Файлы весов `.pt` для этих запусков в исходном проекте не найдены. Точные настройки старых запусков не полностью записаны в JSON; текущий код мог меняться после их получения.

В `pred_oulad.py` положительный класс объединяет `Fail` и `Withdrawn`: это неуспешное завершение обучения. В `experiments/hybrid-attention` используется отдельный протокол (без Withdrawn, прогноз успеха). Метрики двух протоколов нельзя объединять в одну сравнительную таблицу без согласования выборки, целевой метки и разбиения. Подробности исторического протокола находятся в README эксперимента.

## Запуск из корня репозитория

Python 3.12:

```powershell
rtk proxy py -3.12 -m pip install -r requirements.txt
rtk proxy py -3.12 -m models.pred_oulad --help
rtk proxy py -3.12 -m models.pred_oulad --architecture bilstm_attention --data-dir data/oulad/raw --build --tag new_run
rtk proxy py -3.12 -m pytest tests -q
```

Для ансамбля: `--architecture ensemble`. Для LSTM: `--architecture lstm`. Для TCN: `--architecture tcn_bilstm_attention`. Указывайте новый `--tag`, чтобы сохранить исторические результаты.

Для сравнительных экспериментов установите `requirements-experiments.txt`, перейдите в `experiments/hybrid-attention` и запустите `main.py`. В `config.py` доступны переключатели `RUN_BASELINE_ML`, `RUN_KFOLD`, `RUN_ABLATION`. По умолчанию дорогие блоки отключены; включите нужные для повторного сравнения. Повторный запуск может перезаписать таблицы и графики эксперимента.

Исходные данные: `data/oulad/raw`. Подготовленные таблицы уже находятся в `data/oulad/processed`, разбиения — в `data/oulad/splits`. Они доступны локально, но не попадут в клон Git. Для нового клона поместите CSV OULAD в указанную папку и выполните:

```powershell
rtk proxy py -3.12 -m data.oulad_loader --data-dir data/oulad/raw --build
```

Репозиторий создан локально. Публикация на GitHub не выполнялась.
