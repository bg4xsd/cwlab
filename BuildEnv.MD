## **Prepare the python env**

1. Create virtual env by use miniconda
   
   a)  Create env
   
    `conda create  -n cwlab python=3.9`
   
   `conda activate cwlab`
   
   b)  Install basic packages
   
   `conda install scipy numpy matplotlib pandas ipython ipykernel scikit-learn scikit-learn-intelex statsmodels patsy seaborn beautifulsoup4 nodejs simplejson bokeh psutil pylint flake8 yapf autopep8 black requests lxml `
   
   c)  Some Neural Network Libs: 
   
   ` conda install tensorflow-gpu pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia`
   
   d) If you'd like to do some test work by using Jupyter, install them:
   
   `conda install jupyter notebook jupyterlab`

## **Exprot/Import the python env**

2. Using conda
   
   1. Export env
      
         ` conda list -e > requirements.txt`
   
   2. Import env
      
         ` conda install --yes --file requirements.txt`

3. Using pip
   
   If you like to use pip, try pipreqs that can create an simple list of your project. Run below command in your project directory:
   
   ` pip install pipreqs`
   
   1. Export env
      
      `   pip freeze > requirements.txt `
      
      or
      
      `   pipreqs ./ --encoding=utf8`

2. Import evn
   
   `pip install -r requirements.txt`
   Or,  using another source that faster
   `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
