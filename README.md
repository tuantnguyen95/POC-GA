# booster-ai-service

The AI service supports finding UI element, assertions...

### [AI Service](./src/service)

#### Requirements

* Python 3.7.3
* [Docker](https://www.docker.com/products/docker-desktop)
* tesseract 4.0.0
```bash
brew install tesseract@4.0.0
```
* (macOS only) lightgbm >= 3.3.1:
```bash
brew install lightgbm
```

[Optional step] Configure Google Vision instead of tesseract to extract OCR text
* Download the existing file on [S3](https://kobiton-devvn.s3.ap-southeast-1.amazonaws.com/booster-config/googlevision-config.json) or [GG Cloud Console](https://console.cloud.google.com/apis/credentials) (Using tech@kobiton.com account. See more in LastPass).
```bash
export GOOGLE_APPLICATION_CREDENTIALS='path/to/googlevision_credential_json_file'
```
macOS only, export a variable environment to bypass the [known issue](https://github.com/microsoft/LightGBM/issues/4707) of LightGBM library
```bash
export OMP_THREAD_LIMIT=1
```

#### Commands
_Note_: You need to login npm (Get account information in LastPass) in order to install Kobiton private modules in your terminal first.

Download the visual model at [S3 folder](https://kobiton-devvn.s3.ap-southeast-1.amazonaws.com/AI/visual_model/0002/) and put it at [src/service/visual_models](src/service/visual_models)

```bash
yarn
yarn update-schema
yarn build
yarn run-tfserving-docker # start Tensorflow serving
yarn run-service # start AI service
yarn dockerize
```

### [Training Model](./src/train)
#### Create dataset
Kobiton created tools to support on crawling data from Kobiton's devices. The tools are at [kobiton-tool](https://github.com/kobiton/tools).
There are 2 steps:
1. Crawl data by using [random_click_crawler](https://github.com/kobiton/tools/tree/master/random-click-crawler) or [matched-elements-crawler](https://github.com/kobiton/tools/tree/master/matched-elements-crawler).
2. Annotate and export raw data by using 'human_in_loop' from [random_click_crawler](https://github.com/kobiton/tools/tree/master/random-click-crawler).

_Note_: The data from `matched-elements-crawler` hasn't been supported to export for the training yet.

#### Run the training script

- Change path and desired values at [config.py](src/train/configs/config.py).
- Run training:

```bash
yarn run-training
```

#### What Beautiful Is model
1. Generate dataset by using tool [what-beautiful-is](https://github.com/kobiton/tools/tree/master/what-beautiful-is).
2. Run below command to open Jupyter notebook UI in default browser:
_Install Jupyter if you don't have: pip install notebook_
```bash
jupyter notebook
```
3. Open the file [train.ipynb](./src/train/wbi/train.ipynb) in Jupyter notebook UI.
4. Update the path of output data from step 1 in Jupyter file and then run it.


### [Benchmarking](./src/service/benchmark/)

### Get average time for one request to the TF serving service (ex: average over 20)

```bash
cd src && python -m service.benchmark.benchmark_tfserving 127.0.0.1 8500 20
```

### Get average time for one request to the ai service (ex: average over 20)

```bash
cd src && python -m service.benchmark.benchmark_ai_service http://127.0.0.1:5000/element_finding?session_id=1234 20
```

_Note_:

On MacOS, it may be necessary to use the environment variable `OBJC_DISABLE_INITIALIZE_FORK_SAFETY`

```bash
cd src && OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m service.benchmark.benchmark_ai_service http://127.0.0.1:5000/element_finding?session_id=1234 20
```

### [Unit Test](src/service/test)

### element_finding endpoint

- Run AI service:

```bash
yarn build
KOBITON_AI_TF_SERVING_HOST=127.0.0.1 yarn run-service
```

- Run the unit tests in another terminal:

```bash
cd src

# Android
python -m service.test.android.test_service find http://127.0.0.1:5000/element_finding
python -m service.test.android.test_service find_key_in_keyboard http://127.0.0.1:5000/element_finding
python -m service.test.android.test_service compare http://127.0.0.1:5000/element_comparison
python -m service.test.android.test_service text_assertion http://127.0.0.1:5000/text_assertion
python -m service.test.android.test_service text_assertion_w_prime_element_xpaths http://127.0.0.1:5000/text_assertion
python -m service.test.android.test_service visual_assert http://127.0.0.1:5000/visual_verification
python -m service.test.android.test_service visual_assert_w_submission http://127.0.0.1:5000/visual_verification
python -m service.test.android.test_service same_screen http://127.0.0.1:5000/device_same_size
python -m service.test.android.test_service rec_fontsize http://127.0.0.1:5000/recommend/fontsize
python -m service.test.android.test_service accessibility_assert http://127.0.0.1:5000/accessibility_assertion
python -m service.test.android.test_service wbi http://127.0.0.1:5000/wbi

# iOS
python -m service.test.ios.test_service find http://127.0.0.1:5000/element_finding
python -m service.test.ios.test_service compare http://127.0.0.1:5000/element_comparison
python -m service.test.ios.test_service text_assertion http://127.0.0.1:5000/text_assertion
python -m service.test.ios.test_service text_assertion_w_submission http://127.0.0.1:5000/text_assertion
python -m service.test.ios.test_service visual_assert http://127.0.0.1:5000/visual_verification
python -m service.test.ios.test_service find_webview http://127.0.0.1:5000/element_finding

# WebView
python -m service.test.webview.test_service find http://127.0.0.1:5000/element_finding
python -m service.test.webview.test_service find_html http://127.0.0.1:5000/element_finding
python -m service.test.webview.test_service compare http://127.0.0.1:5000/element_comparison
python -m service.test.webview.test_service text_assertion http://127.0.0.1:5000/text_assertion

# Call Luna Android
python -m service.test.android.test_service find http://127.0.0.1:5000/luna/element_finding
# Call Luna iOS
python -m service.test.ios.test_service find http://127.0.0.1:5000/luna/element_finding

# Call Facenet Android
python -m service.test.android.test_service find http://127.0.0.1:5000/facenet/element_finding
# Call Facenet iOS
python -m service.test.ios.test_service find http://127.0.0.1:5000/facenet/element_finding

# Run all unit tests
python -m unittest service.test.android.test_service

# Run a specific unit test method
python -m unittest service.test.android.test_service.AIServiceTestCase.test_finding_element_by_image

### Application Metrics

The application metric endpoint is at /metrics. See [here](src/service/metrics.py) for list of metrics.