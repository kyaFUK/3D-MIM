FROM tensorflow/tensorflow:2.4.1-gpu

LABEL version="1.0"
LABEL description="Container for running MIM using GPU"

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN apt-get install -y libgl1-mesa-dev
RUN pip install pillow
RUN pip install pandas
RUN pip install scikit-image

COPY $(pwd) ~/home/MIM-master