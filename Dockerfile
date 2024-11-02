# Select image matching your hardware from https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

WORKDIR /job-offers-classifier
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential gcc-8 g++-8

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

RUN gcc --version

RUN python3 -c "import sys; print(sys.version)"

RUN pip3 install -r requirements.txt

CMD python3 main.py fit LinearJobOffersClassifier -x train_test_data/example/x_train.txt -y train_test_data/example/y_train.txt -h train_test_data/example/classes.tsv -m models/main_example
