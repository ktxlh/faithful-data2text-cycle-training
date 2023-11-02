# Creat new conda venv
conda create --name nlpproj python=3.11
conda activate nlpproj

# Install torch compatible with your cuda version first!
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy transformers nltk tqdm ipykernel ipywidgets matplotlib

# Install nltk punkt for word_tokenize
python -c "import nltk;nltk.download('punkt')"
