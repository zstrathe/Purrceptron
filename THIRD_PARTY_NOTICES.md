# Third-Party Notices

Purrceptron is licensed under the [Purrceptron Source-Available License 1.0](LICENSE.md). The portions listed below are based on third-party work and remain under their own licenses. The Purrceptron license does not restrict any right those licenses grant for those portions.

## hailo-apps

- Source: https://github.com/hailo-ai/hailo-apps
- License: MIT
- Used in:
  - `deploy/detect.py`: the detection callback structure (the `app_callback_class` subclass, the `app_callback` function that reads detections and tracking IDs from the buffer, and running `GStreamerDetectionApp`) follows the hailo-apps detection example.
  - `deploy/coco_labels.json`: uses the hailo-apps labels-file format.

```
MIT License

Copyright (c) 2024 Hailo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies installed separately

These are not included in this repository. They are installed by the setup scripts and are covered by their own licenses:

- PyTorch and torchvision (BSD-3-Clause), including the pretrained weights torchvision downloads.
- The Hailo runtime, hailo-apps and Hailo model files on the Raspberry Pi.
- The other Python packages listed in `requirements-*.txt`.
