docker run --gpus all -d --mount type=bind,src=/"$(pwd)/llmprototypes",target=/llmprototypes -it --name $1 $2
