# GPU baseline sweep results

| Backbone | Freeze mode | Seed | Best val accuracy | Epochs | Seconds per epoch | Run |
|---|---|---:|---:|---:|---:|---|
| resnet50 | frozen | 42 | 0.995799 | 10 | 2.23 | `runs\sweeps\gpu_aug\resnet50__frozen__9e45293166\seed_42\20261027-121237-099733-resnet50` |
| resnet50 | last_block | 42 | 0.995399 | 6 | 147.56 | `runs\sweeps\gpu_aug\resnet50__last_block__c6ce901214\seed_42\20261027-121701-257745-resnet50` |
| resnet50 | full | 42 | 0.993599 | 5 | 401.93 | `runs\sweeps\gpu_aug\resnet50__full__f987faee5e\seed_42\20261027-123314-168759-resnet50` |
| resnet18 | last_block | 42 | 0.988598 | 6 | 120.57 | `runs\sweeps\gpu_aug\resnet18__last_block__b85a42fdf3\seed_42\20261027-114502-285114-resnet18` |
| resnet18 | frozen | 42 | 0.987998 | 10 | 2.10 | `runs\sweeps\gpu_aug\resnet18__frozen__1eda6c6e71\seed_42\20261027-114158-973797-resnet18` |
| resnet18 | full | 42 | 0.987598 | 5 | 167.64 | `runs\sweeps\gpu_aug\resnet18__full__a492cb266c\seed_42\20261027-115753-652168-resnet18` |

The **resnet50 / frozen** configuration had the top validation accuracy (0.995799). It is selected as the baseline because it achieved the best validation result and had a lower compute cost than the fine-tuned configurations; this keeps later comparisons efficient without giving up measured validation accuracy.
