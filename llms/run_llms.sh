python runnables/scattergories.py --model qwen --language english --strategy 0 --letters c
python runnables/scattergories.py --model qwen --language english --strategy 1 --letters c
python runnables/scattergories.py --model qwen --language german --strategy 0 --letters c
python runnables/scattergories.py --model qwen --language german --strategy 1 --letters c
python runnables/scattergories.py --model qwen --language spanish --strategy 0 --letters c
python runnables/scattergories.py --model qwen --language spanish --strategy 1 --letters c
hf cache rm model/Qwen/Qwen2.5-7B-Instruct -y

python runnables/scattergories.py --model llama --language english --strategy 0 --letters c
python runnables/scattergories.py --model llama --language english --strategy 1 --letters c
python runnables/scattergories.py --model llama --language german --strategy 0 --letters c  
python runnables/scattergories.py --model llama --language german --strategy 1 --letters c  
python runnables/scattergories.py --model llama --language spanish --strategy 0 --letters c  
python runnables/scattergories.py --model llama --language spanish --strategy 1 --letters c  
hf cache rm model/meta-llama/Meta-Llama-3.1-8B-Instruct -y

python runnables/scattergories.py --model mistral --language english --strategy 0 --letters c  
python runnables/scattergories.py --model mistral --language english --strategy 1 --letters c  
python runnables/scattergories.py --model mistral --language german --strategy 0 --letters c  
python runnables/scattergories.py --model mistral --language german --strategy 1 --letters c  
python runnables/scattergories.py --model mistral --language spanish --strategy 0 --letters c  
python runnables/scattergories.py --model mistral --language spanish --strategy 1 --letters c  
hf cache rm model/mistralai/Mistral-7B-Instruct-v0.3 -y

python runnables/scattergories.py --model phi --language english --strategy 0 --letters c  
python runnables/scattergories.py --model phi --language english --strategy 1 --letters c  
python runnables/scattergories.py --model phi --language german --strategy 0 --letters c  
python runnables/scattergories.py --model phi --language german --strategy 1 --letters c  
python runnables/scattergories.py --model phi --language spanish --strategy 0 --letters c  
python runnables/scattergories.py --model phi --language spanish --strategy 1 --letters c  
hf cache rm model/microsoft/Phi-3-mini-4k-instruct -y