# Contributing

Contributions are welcome: bug fixes, improvements, documentation, and new features.

This project is **source-available, not open source**. It is licensed under the [Purrceptron Source-Available License 1.0](LICENSE.md). In short:

- You may use it for noncommercial purposes, and change it privately for your own setup.
- You may fork it **only** to prepare contributions back to this repository.
- You may not redistribute it, publish a modified version, or run a fork as a separate project.
- You may not use it, or any part of it, as data for training, evaluating, or retrieval by any AI or machine learning model, or give it to any AI service whose provider is allowed to train on it. AI coding assistants are fine if their terms don't let the provider train on your input.
- Any model you train with it may be used only by you, for noncommercial purposes. You may not share, sell, or commercially use it.

## How to contribute

1. Fork the repository and create a branch for your change.
2. Make your change. Keep pull requests focused on one thing.
3. Do not commit local-only files. These are gitignored: `configs/project.yaml`, `.env`, and `deploy/labels.json`. If you add or change config options, update `configs/project.yaml.example` instead.
4. Do not commit captured images, datasets, trained models, or virtual environments.
5. Open a pull request that describes what you changed and how you tested it (for example, on a Raspberry Pi with the Hailo accelerator, or in the training pipeline).
6. In the pull request, check the box confirming that you agree to the contribution license below.

For larger changes, consider opening an issue first to discuss the approach.

## Contribution license

By submitting a contribution (for example, a pull request, patch, or suggested code) to this repository, you agree that:

1. The contribution is your own original work, or you otherwise have the right to submit it under these terms. If it includes work by others, you say so in the pull request, and that work's license must allow it to be included under these terms.
2. You grant Zack Strathe a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license, under your copyright and under any patent claims you can license that your contribution would otherwise infringe, to use, copy, modify, sublicense, and distribute your contribution, and to make, use, sell, and otherwise transfer anything that includes it, under any terms, including the project's license and any separate commercial license.
3. You keep the copyright to your contribution, and you are free to use your own contribution elsewhere.

You confirm this agreement by checking the box in the pull request template. Pull requests without that confirmation will not be merged. If you are not willing to agree to these terms, please do not submit a contribution.
