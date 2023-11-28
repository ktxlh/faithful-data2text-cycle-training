# Creat new conda venv
conda create --name cycletrain python=3.11
conda activate cycletrain

# Install torch compatible with your cuda version first!
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge unidecode
conda install numpy pandas transformers datasets sentencepiece
pip install nltk tqdm ipykernel ipywidgets matplotlib bert_score absl-py rouge_score

# Install nltk punkt for word_tokenize
python -c "import nltk;nltk.download('punkt')"
