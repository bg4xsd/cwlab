## **Prepare the python env**

Create virtual env by use miniconda

1) Create env

   `conda create  -n cwlab python=3.9`

`conda activate cwlab`

2. Install basic packages

`conda install scipy numpy matplotlib pandas ipython ipykernel scikit-learn scikit-learn-intelex statsmodels patsy seaborn beautifulsoup4 nodejs simplejson bokeh psutil pylint flake8 yapf autopep8 black requests lxml`

3. Some Neural Network Libs:

`conda install tensorflow-gpu pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia`

4. If you'd like to do some test work by using Jupyter, install them:

`conda install jupyter notebook jupyterlab`

## **Exprot/Import the python env**

1. Using conda

   a) Export env

      `conda list -e > requirements.txt`

   b) Import env

      `conda install --yes --file requirements.txt`

2. Using pip

   If you like to use pip, try pipreqs that can create an simple list of your project. Run below command in your project directory:

   `pip install pipreqs`

   a) Export env

   `pip freeze > requirements.txt`

      or

   `pipreqs ./ --encoding=utf8`

   b) Import evn

   `pip install -r requirements.txt`
      Or,  using another source that faster
   `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r  requirements.txt`

Last update by BFcat @ 2022.12.10
