pip install -r requirements.txt

wget -O datasets/llava_instruct_150k.json https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json
wget -O datasets/train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip -q datasets/train2014.zip -d datasets/