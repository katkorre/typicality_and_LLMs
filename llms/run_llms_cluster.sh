python runnables/scattergories.py --model qwen --language english --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model qwen --language english --strategy 1 --letters c --quantize 0
python runnables/scattergories.py --model qwen --language german --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model qwen --language german --strategy 1 --letters c --quantize 0
python runnables/scattergories.py --model qwen --language spanish --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model qwen --language spanish --strategy 1 --letters c --quantize 0
hf cache rm model/Qwen/Qwen2.5-7B-Instruct -y

python runnables/scattergories.py --model gemma --language english --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model gemma --language english --strategy 1 --letters c --quantize 0
python runnables/scattergories.py --model gemma --language german --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model gemma --language german --strategy 1 --letters c --quantize 0
python runnables/scattergories.py --model gemma --language spanish --strategy 0 --letters c --quantize 0
python runnables/scattergories.py --model gemma --language spanish --strategy 1 --letters c --quantize 0
hf cache rm model/google/gemma-3-12b-it -y