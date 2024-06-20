# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# 更新包列表并安装OpenGL库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0

# Set the working directory
WORKDIR /Pao-Ding

# Copy the current directory contents into the container
COPY ./  /Pao-Ding

# Install any needed packages specified in requirements.txt
RUN pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run when the container launches
# CMD ["python", "main.py"]