pip install -r requirements.txt

wget -O datasets/llava_instruct_150k.json https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json
wget -O datasets/train2014.zip http://images.cocodataset.org/zips/train2014.zip
wget -O datasets/dataset_v7w_telling.zip https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip
wget -O datasets/visual7w_images.zip http://vision.stanford.edu/yukezhu/visual7w_images.zip
unzip -q datasets/train2014.zip -d datasets/
unzip -q datasets/visual7w_images.zip -d datasets/
unzip -q datasets/dataset_v7w_telling.zip -d datasets/